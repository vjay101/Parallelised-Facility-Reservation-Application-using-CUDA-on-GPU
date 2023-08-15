	#include <iostream>
	#include <stdio.h>
	#include <cuda.h>

	#define max_N 100000
	#define max_P 30
	#define BLOCKSIZE 1024

	using namespace std;




//****************Kernels start here***************************

//kernel for val_initial the lock array
__global__ void val_initial(int target,int *lock)
	{
    unsigned id =threadIdx.x-1+blockIdx.x*BLOCKSIZE;
		if(id+1<target) lock[id+1]=10000;
	}


	//kernel to calculate the request processed by computer center
	__global__ void assignwork(int start,float rn,int R,int *d_var_lock,float z,int *d_centre,int *d_facility,int *d_offset,int *d_totalcapacity,int volatile *d_totalslots,int *d_neces_cen,int * d_facility_offset,int *d_neces_start,int *d_neces_fac,unsigned int *d_total_succ_reqs,int *d_neces_slots)
	{
 
     	__shared__ unsigned total;
		   int n;
       int p=0;
		   int id=start+threadIdx.x+blockIdx.x*BLOCKSIZE;
		
	

    int total_house;            //variable holding the total no of centers
    total_house=d_neces_cen[id];
    total=1;
    int slot_no;
    slot_no=d_neces_fac[id]*24+d_offset[total_house];
    int choose;
    choose=d_neces_fac[id]+d_facility_offset[total_house];
    int temp=0;
    int perform=1;
  __syncthreads();            //to synchronise all the requests

	
    int c=-1+d_neces_start[id];
    int d=d_neces_slots[id]-2+d_neces_start[id];

			
   
	    do{
      total=1;
	    if(perform!=p) atomicMin(&(d_var_lock[choose]),id);          //to find out the minimum request id and serve it first
      

		__syncthreads();
			
	    if(perform!=0)
			{			
         if(id==d_var_lock[choose])
				{
				for(int i=c+slot_no;i<=(d+slot_no);i++)
					{
				      d_totalslots[i]=-1+d_totalslots[i];
              if(d_totalslots[i]<p)
						{	
							n = i;
							temp = 1;
							break;						
						}
					}
					
					if(temp!=0)
					{

			         for(int i = (c+slot_no); i <= n; i++)
						{
							d_totalslots[i]=1+d_totalslots[i];
						}
						temp = p;
					}
					else
					{	   int val=1+total_house;
				        atomicInc(&d_total_succ_reqs[val],R);
			            atomicInc(&d_total_succ_reqs[0],R);
			        }
					
					
					perform=p;
					d_var_lock[choose] = 1000000;	
				}

				else if(perform!=0) total=p;
					
			}
			__syncthreads();
					
		}while(total==p);
		__syncthreads();
		
	}

//*********************Kernels end here**************************


int main(int argc,char **argv)
{
// variable declarations...
    int N,*total_house,*facility,*capacity,*fac_ids, *succ_reqs, *tot_reqs;
    

    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &N ); // N is number of centres
	
    // Allocate memory on cpu
    total_house=(int*)malloc(N * sizeof (int));  // Computer  total_house numbers
    facility=(int*)malloc(N * sizeof (int));  // Number of facilities in each computer total_house
    fac_ids=(int*)malloc(max_P * N  * sizeof (int));  // Facility room numbers of each computer total_house
    capacity=(int*)malloc(max_P * N * sizeof (int));  // stores capacities of each facility for every computer total_house 


    int success=0;  // total successful requests
    int fail = 0;   // total failed requests
    tot_reqs = (int *)malloc(N*sizeof(int));   // total requests for each total_house
    succ_reqs = (int *)malloc((N+1)*sizeof(int)); // total successful requests for each total_house

    // Input the computer centres data
    int k1=0 , k2 = 0;
    for(int i=0;i<N;i++)
    {
      fscanf( inputfilepointer, "%d", &total_house[i] );
      fscanf( inputfilepointer, "%d", &facility[i] );
      
      for(int j=0;j<facility[i];j++)
      {
        fscanf( inputfilepointer, "%d", &fac_ids[k1] );
        k1++;
      }
      for(int j=0;j<facility[i];j++)
      {
        fscanf( inputfilepointer, "%d", &capacity[k2]);
        k2++;     
      }
    }

    // variable declarations
    int *req_id, *req_cen, *req_fac, *req_start, *req_slots;   // Number of total_slots requested for every request
    
    // Allocate memory on CPU 
	  int R;
	  fscanf( inputfilepointer, "%d", &R); // Total requests
    req_id = (int *) malloc ( (R) * sizeof (int) );  // Request ids
    req_cen = (int *) malloc ( (R) * sizeof (int) );  // Requested computer total_house
    req_fac = (int *) malloc ( (R) * sizeof (int) );  // Requested facility
    req_start = (int *) malloc ( (R) * sizeof (int) );  // Start slot of every request
    req_slots = (int *) malloc ( (R) * sizeof (int) );   // Number of total_slots requested for every request
    
    // Input the user request data
    for(int j = 0; j < R; j++)
    {
       fscanf( inputfilepointer, "%d", &req_id[j]);
       fscanf( inputfilepointer, "%d", &req_cen[j]);
       fscanf( inputfilepointer, "%d", &req_fac[j]);
       fscanf( inputfilepointer, "%d", &req_start[j]);
       fscanf( inputfilepointer, "%d", &req_slots[j]);
       tot_reqs[req_cen[j]]+=1;  
    }


	//*************************Kernel calling start**************************************************
 
    int temp1=0;      //temporary variables used
    int temp2=0;
    int *total_slots;
    total_slots=(int*)malloc(max_P*N*(24)*sizeof(int));
	  int *facility_no;
	  facility_no=(int*)malloc(N*sizeof(int));
	  int *off_value;
    off_value=(int*)malloc(N*sizeof(int));
	 
  	unsigned int *d_total_succ_reqs;       //to maintain the total succ request at device
	
     //memory allocation at GPU for all the required variables
    int *d_totalslots;
		cudaMalloc(&d_totalslots,(max_P*N*24)*sizeof(int));
		int *d_totalcapacity;
		cudaMalloc(&d_totalcapacity,(max_P*N)*sizeof(int));
    int *d_facility_offset;
		cudaMalloc(&d_facility_offset,(max_P*N)*sizeof(int));
		int *d_total_off;
		cudaMalloc(&d_total_off,(N)*sizeof(int));	
		int *d_centre;
		cudaMalloc(&d_centre,(N)*sizeof(int));
		int *d_facility;
		cudaMalloc(&d_facility,(N)*sizeof(int));
	
		int *d_var_lock;
	  int i=0;
    while(i<N)
          {
             off_value[i]=temp2;
		         facility_no[i]=temp1;
		  
      for(int j=0;j<facility[i];j++)
		  { int k=temp2;
		    while(k<temp2+24){
			  total_slots[k]=capacity[temp1];
			  k++;
		     }
		  temp2=temp2+24;
		  temp1=temp1+1;}
		i++;
		}
    // Copy memory from host to device for all the required variables
		cudaMemcpy(d_totalslots,total_slots,temp2*sizeof(int),cudaMemcpyHostToDevice);
	  cudaMemcpy(d_total_off,off_value,N*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_facility_offset,facility_no,temp2*sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(d_facility,facility,N*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_totalcapacity,capacity,temp1*sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(d_centre,total_house,N*sizeof(int),cudaMemcpyHostToDevice);
		
    
		int initialiseblock=ceil((float(temp1)/BLOCKSIZE));
		
		cudaMalloc(&d_var_lock,(temp1)*sizeof(int));
    //kernel calling to initialize the lock array
		val_initial<<<initialiseblock,BLOCKSIZE>>>(temp1,d_var_lock);	
    
        int totalblocks=R/BLOCKSIZE;     //calculating total no of blocks required
        int *d_neces_fac;
        cudaMalloc(&d_neces_fac,(R)*sizeof(int));
		    cudaMemcpy(d_neces_fac,req_fac,R*sizeof(int),cudaMemcpyHostToDevice);
			
     //memory allocation to device variables and copying memory
      int *d_neces_cen;
			cudaMalloc(&d_neces_cen,(R)*sizeof(int));
			cudaMemcpy(d_neces_cen,req_cen,R*sizeof(int),cudaMemcpyHostToDevice);
            
			int *d_neces_start;
			cudaMalloc(&d_neces_start,(R)*sizeof(int));
			cudaMemcpy(d_neces_start,req_start,R*sizeof(int),cudaMemcpyHostToDevice);
 
      int *d_neces_slots;
			cudaMalloc(&d_neces_slots,(R)*sizeof(int));
			cudaMemcpy(d_neces_slots,req_slots,R*sizeof(int),cudaMemcpyHostToDevice);

			cudaMalloc(&d_total_succ_reqs,(N+1)*sizeof(int));
			cudaMemcpy(d_total_succ_reqs,succ_reqs,(N+1)* sizeof(int),cudaMemcpyHostToDevice);
			
			
      int count=0;   //maintains the no of kernel calling
			int extrathread=R%BLOCKSIZE;
      
			for(int x=totalblocks;x>0;x--){
              count++;
            assignwork<<<1,BLOCKSIZE>>>(BLOCKSIZE*(totalblocks-x),0,R,d_var_lock,0,d_centre,d_facility,d_total_off,d_totalcapacity,d_totalslots,d_neces_cen,d_facility_offset,d_neces_start,d_neces_fac,d_total_succ_reqs,d_neces_slots);
				   count=count+x;
         cudaDeviceSynchronize();
			}
			

       //After counting count,1 more time calling assignwork required to complete the required task
			assignwork<<<1,extrathread>>>(BLOCKSIZE*(totalblocks),0,R,d_var_lock,0,d_centre,d_facility,d_total_off,d_totalcapacity,d_totalslots,d_neces_cen,d_facility_offset,d_neces_start,d_neces_fac,d_total_succ_reqs,d_neces_slots);
      count++;		
			cudaMemcpy(succ_reqs,d_total_succ_reqs,(N+1)*sizeof(int),cudaMemcpyDeviceToHost);

    
   
   //*****************************Kernel calling end***************************************************



    // Output
    char *outputfilename = argv[2]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    fprintf( outputfilepointer, "%d %d\n",succ_reqs[0], R-succ_reqs[0]);
    for(int j = 0; j < N; j++)
    {
        fprintf( outputfilepointer, "%d %d\n", succ_reqs[j+1], tot_reqs[j]-succ_reqs[j+1]);
    }
    fclose( inputfilepointer );
    fclose( outputfilepointer );
    cudaDeviceSynchronize();
	  return 0;
			
	}
