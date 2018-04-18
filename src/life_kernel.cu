
// Reads a cell at (x+dx, y+dy)
// Reads a cell in GPU memory
__device__ int read_cell(int * source_domain, int x, int y, int dx, int dy, unsigned int domain_x, unsigned int domain_y)
{
    x = (unsigned int)(x + dx) % domain_x;	// Wrap around
    y = (unsigned int)(y + dy) % domain_y;
    return source_domain[y * domain_x + x];
}

__device__ void write_cell(int * source_domain, int x, int y, int dx, int dy, unsigned int domain_x, unsigned int domain_y, int value)
{
    x = (unsigned int)(x + dx) % domain_x;	// Wrap around
    y = (unsigned int)(y + dy) % domain_y;
    source_domain[y * domain_x + x] = value;
}

__device__ void update(int * source_domain, int x, int y, int dx, int dy, unsigned int domain_x, unsigned int domain_y, int *nn, int *n1, int *n2)
{
	unsigned int cell = read_cell(source_domain,x,y,dx,dy,domain_x,domain_y);
	if (cell != 0)
	{
		(*nn)++;
		if (cell == 1)
			(*n1)++;
		else
			(*n2)++;
	}
}

__device__ void neighbors(int * source_domain, int x, int y, unsigned int domain_x, unsigned int domain_y, int *nn, int *n1, int *n2)
{
	int dx,dy;

	(*nn) = 0; (*n1) = 0; (*n2) = 0;

	// same line
	dx = -1; dy = 0; update(source_domain,x,y,dx,dy,domain_x,domain_y,nn,n1,n2);
	dx = +1; dy = 0; update(source_domain,x,y,dx,dy,domain_x,domain_y,nn,n1,n2);

	// one line down
	dx = -1; dy = +1; update(source_domain,x,y,dx,dy,domain_x,domain_y,nn,n1,n2);
	dx =  0; dy = +1; update(source_domain,x,y,dx,dy,domain_x,domain_y,nn,n1,n2);
	dx = +1; dy = +1; update(source_domain,x,y,dx,dy,domain_x,domain_y,nn,n1,n2);

	// one line up
	dx = -1; dy = -1; update(source_domain,x,y,dx,dy,domain_x,domain_y,nn,n1,n2);
	dx =  0; dy = -1; update(source_domain,x,y,dx,dy,domain_x,domain_y,nn,n1,n2);
	dx = +1; dy = -1; update(source_domain,x,y,dx,dy,domain_x,domain_y,nn,n1,n2);
}

// Compute kernel
__global__ void life_kernel(int * source_domain, int * dest_domain, int domain_x, int domain_y)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y;
	
	// position de la cellule traité dans le domain couvert par le block
	int x = threadIdx.x;
	int y = threadIdx.y;
	
	// domain couvert par le block (shared memory)
	int s_domain_x = blockDim.x;
	int s_domain_y = blockDim.y;
	
	extern __shared__ int shared_domain[];
    
    int nn,n1,n2;
    
    // Read cell
    int myself = read_cell(source_domain, tx, ty, 0, 0, domain_x, domain_y);
	// Ecriture à l'intérieur des bords (+1 et +2)
	write_cell(shared_domain, x+1, y+1, 0, 0, s_domain_x+2, s_domain_y+2, myself);
	// Remplissage des bordures
	if(x == 0)
		write_cell(shared_domain, 0, y+1, 0, 0, s_domain_x+2, s_domain_y+2, read_cell(source_domain, tx, ty, -1, 0, domain_x, domain_y))
	if(y == 0)
		write_cell(shared_domain, x+1, 0, 0, 0, s_domain_x+2, s_domain_y+2, read_cell(source_domain, tx, ty, 0, -1, domain_x, domain_y))
	if(x == 0 && y == 0)
		write_cell(shared_domain, 0, 0, 0, 0, s_domain_x+2, s_domain_y+2, read_cell(source_domain, tx, ty, -1, -1, domain_x, domain_y))
	if(x == s_domain_x)
		write_cell(shared_domain, x+1+1, y+1, 0, 0, s_domain_x+2, s_domain_y+2, read_cell(source_domain, tx, ty, +1, 0, domain_x, domain_y))
	if(y == s_domain_y)
		write_cell(shared_domain, x+1, y+1+1, 0, 0, s_domain_x+2, s_domain_y+2, read_cell(source_domain, tx, ty, 0, +1, domain_x, domain_y))
	if(x == s_domain_x && y == s_domain_y)
		write_cell(shared_domain, x+1+1, y+1+1, 0, 0, s_domain_x+2, s_domain_y+2, read_cell(source_domain, tx, ty, +1, +1, domain_x, domain_y))
    
	neighbors(shared_domain, x+1, y+1, s_domain_x+2, s_domain_y+2, &nn, &n1, &n2);

	if(myself != 0)
		// Wrong number of neighbors, the cell dies
		if(!((nn > 1) && (nn < 4)))
			myself = 0;
	// Reproduction, a new cell is born
	else if(nn == 3)
	{
		// Takes on the dominant genus
		if(n1 >= 2)
			myself = 1;
		else
			myself = 2;
		change++;
	}
	
	write_cell(dest_domain, tx, ty, 0, 0, domain_x, domain_y, myself);
}

