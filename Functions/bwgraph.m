function G = bwgraph( bw, Options )
%BWGRAPH Create a graph of connected pixels in 2D images or 3D volumes.
%  BWGRAPH can be used to find the shortest path between 2 pixels.
%  G = bwgraph( bw, Name=Value )
% 
%  INPUTS
%  - bw                    Binary image, given as a 2D or 3D, numeric or 
%                          logical array. For numeric input, non-zero 
%                          pixels are considered to be 1 (true).
%  - Name-Value Arguments
%    + Connectivity        Pixel connectivity, defining whether pixels are 
%                          connected when they share either a face, edge, 
%                          or corner. For a 2D bw, valid values are 4 
%                          (edge) and 8 (corner), and in 3D, 6 (face), 18 
%                          (edge) and 26 (corner). Default is the maximal 
%                          connectivity, i.e., 8 or 26.
%    + NodeWeights         Node weights, given for each element of bw. For
%                          example, the node weight could represent the 
%                          distance between that pixel and the nearest 
%                          non-zero pixel of bw, as calculated with bwdist.
%                          Edge weights for the graph are set as the mean
%                          node weight of the two connected pixels. Numeric
%                          array with the same size as bw.
%  OUTPUT
%  - G                     graph object with nodes for all pixels of bw and
%                          edges between connected non-zero pixels. By 
%                          default, edge weights are the Euclidean distance
%                          between the centers of connected pixels. The 
%                          indices of the graph's nodes are equivalent to 
%                          the linear indices of the pixels in bw. 
%                          Therefore, use sub2ind and ind2sub to convert 
%                          between pixel subscripts and node indices.
% 
%   EXAMPLES: Please see the file 'examples.mlx' or 'examples.pdf'.
%    
%   Created in 2022b. Compatible with 2019b and later. Compatible with all 
%   platforms. Please cite George Abrahams 
%   https://github.com/WD40andTape/bwgraph.
% 
%   See also GRAPH, SHORTESTPATH, DISTANCES, BWDIST, CONNDEF

%   Part of this code was inspired by findNeighbours by Patrick Granton 
%   and improved by Ahmed Zankoor.
%   https://mathworks.com/matlabcentral/fileexchange/68549-findneighbours
% 
%   Published under MIT License (see LICENSE.txt).
%   Copyright (c) 2023 George Abrahams.
%   - https://github.com/WD40andTape/
%   - https://www.linkedin.com/in/georgeabrahams/

    arguments
        bw logical { mustBeNonempty, mustBe2Dor3D }
        Options.NodeWeights double ...
            { mustBeSameSize( bw, Options.NodeWeights ) }
        Options.Connectivity (1,1) uint8 { mustBeValidConnectivity( ...
            bw, Options.Connectivity ) } = 3^ndims( bw )-1
    end

    sz = size( bw );
    dim = length( sz ); % 2D or 3D.

    % Create base, the IJ(K) offset to each neighbour as defined by conn.
    connMatrix = images.internal.getBinaryConnectivityMatrix( ...
        Options.Connectivity );
    connMatrix(2,2,dim-1) = 0; % Set the central index to 0.
    [ baseI, baseJ, baseK ] = ndgrid( -1 : 1 );
    % Apply a mask so that edges aren't calculated twice, i.e., once in 
    % each direction.
    if dim == 2
        mask = logical( [0 0 1; 0 0 1; 0 1 1] );
        connMatrix = connMatrix & mask;
        base = [ baseI(connMatrix), baseJ(connMatrix) ];
    else
        mask = false( 3, 3, 3 );
        mask(:,:,1) = [ 0 0 1; 0 0 1; 0 1 1 ];
        mask(:,:,2) = [ 0 0 1; 0 0 1; 0 1 1 ];
        mask(:,:,3) = [ 0 0 1; 0 1 1; 0 1 1 ];
        connMatrix = connMatrix & mask;
        base = [ baseI(connMatrix), baseJ(connMatrix), baseK(connMatrix) ];
    end
    conn = sum( connMatrix, 'all' );

    % Find IJ(K) indices of all non-zero elements of bw.
    if dim == 2
        [ I1, I2 ] = ind2sub( sz, find( bw ) );
        source = [ I1, I2 ];
    else
        [ I1, I2, I3 ] = ind2sub( sz, find( bw ) );
        source = [ I1, I2, I3 ];
    end
    % Clear temporary variables to save memory.
    clearvars I1 I2 I3
    % Add base indices to source indices to find all neighbours.
    n = size( source, 1 );
    source = repelem( source, conn, 1 );
    neighbours = source + repmat( base, n, 1 );
    % Remove invalid neighbours, i.e., those beyond the matrix boundaries.
    valid = all( neighbours > 0 & neighbours <= sz, 2 );
    source = source(valid,:);
    neighbours = neighbours(valid,:);
    % Calculate the Euclidean distances from source to neighbour.
    if ~isfield( Options, 'NodeWeights' )
        weights = vecnorm( base, 2, 2 );
        weights = repmat( weights, n, 1 );
        weights = weights(valid,:);
    end
    
    % Convert IJ(K) indices to linear indices.
    if dim == 2
        source = sub2ind( sz, source(:,1), source(:,2) );
        neighbours = sub2ind( sz, neighbours(:,1), neighbours(:,2) );
    else
        source =  sub2ind( sz, source(:,1), source(:,2), source(:,3) );
        neighbours = sub2ind( sz, ...
            neighbours(:,1), neighbours(:,2), neighbours(:,3) );
    end
    % Remove non-zero neighbours.
    valid = bw(neighbours) ~= 0;
    source = source(valid,:);
    neighbours = neighbours(valid,:);
    % Calculate weights.
    if ~isfield( Options, 'NodeWeights' )
        weights = weights(valid,:);
    else
        % Set edge weights to the average of the connecting node weights.
        weights = mean( Options.NodeWeights( [source neighbours] ), 2 );
    end
    % Clear bw input to save memory, as the graph object construction
    % is memory intensive.
    clearvars bw
    
    % Construct the graph object.
    numNodes = prod( sz );
    G = graph( source, neighbours, double( weights ), numNodes );
    
end

%% Validation functions

function mustBe2Dor3D( a )
    if ndims( a ) > 3 % ndims is always greater than 2.
        id = "bwgraph:Validators:MatrixNot2Dor3D";
        msg = "Must be either 2D or 3D.";
        throwAsCaller( MException( id, msg ) )
    end
end

function mustBeValidConnectivity( a, conn )
    id = "bwgraph:Validators:ConnectivityInvalid";
    if ismatrix( a ) && ~ismember( conn, [4 8] )
        msg = "Valid connectivities for a 2D array are 4 and 8.";
        throwAsCaller( MException( id, msg ) )
    elseif ndims( a ) == 3 && ~ismember( conn, [6 18 26] )
        msg = "Valid connectivities for a 3D array are 6, 18, and 26.";
        throwAsCaller( MException( id, msg ) )
    end
end

function mustBeSameSize( a, b )
    if ~isequal( size( a ), size( b ) )
        id = "bwgraph:Validators:IncompatibleSizeInputs";
        msg = "Must be the same size as bw.";
        throwAsCaller( MException( id, msg ) )
    end
end