#include <stdbool.h>
#include "mpi.h"

#include "ppp/ppp.h"
#include "ppp_pnm/ppp_pnm.h"

struct myBody {
    bool calculated;
    long double aX;
    long double aY;
};

static void compute(body *i, body *j,
                    long double *ax, long double *ay)
{
    long double aij_x, aij_y, dx, dy, r3;
    dx = j->x - i->x;
    dy = j->y - i->y;
    r3 = powl(sqrtl(dx*dx+dy*dy), 3);
    aij_x = dx/r3;
    aij_y = dy/r3;
    *ax = aij_x * j->mass;
    *ay = aij_y * j->mass;
}

static void update(body *b, double deltaT,
                   long double ax, long double ay)
{
    long double dvx, dvy;
    dvx = ax * (G*deltaT);
    dvy = ay * (G*deltaT);
    b->x += (b->vx + dvx/2) * deltaT;
    b->y += (b->vy + dvy/2) * deltaT;
    b->vx += dvx;
    b->vy += dvy;
}

void compute_parallel(struct TaskInput *TI) {
    int np, self;
    const bool debug = TI->debug;
    const long double deltaT = TI->deltaT;
    const int nSteps = TI->nSteps;
    const int imageStep = TI->imageStep;
    const int nBodies = TI->nBodies;
    body *bodies = TI->bodies;
    body *local_bodies, *finalBodies;
    int *recv_counts;
    int* send_counts;
    int* displs, *displs_send;
    struct myBody** allToAll;
    body *recvbuf;
    bool** iTOJ;
    
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &self);

    //bcast the info comes from main
    MPI_Bcast(&nBodies, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&imageStep, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nSteps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&deltaT, 1, MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);

    int local_nBodies = nBodies / np;
    int local_nBodies_woRemainder = local_nBodies;  // number of bodies each process if there was not a remainder
    //printf("The total num  of bodies is: %d\n", nBodies);

    int remainder;
    remainder = nBodies % np;

    if (self < remainder) {
        local_nBodies++;
    }

    long double accelsX[local_nBodies];
    long double accelsY[local_nBodies];

    send_counts = malloc(local_nBodies * sizeof(int));
    displs_send = malloc(local_nBodies * sizeof(int));
    local_bodies = malloc(local_nBodies * sizeof(body));

    /* Fill in the displacements and send_counts */
    for (int i=0; i<np; i++) {
        send_counts[i] = local_nBodies_woRemainder;
        if (i < remainder) {    //there is remainder and the process will have one remaining
            send_counts[i]++;
            displs_send[i] = i * (local_nBodies_woRemainder +1);
        }
        else if (i != 0) {  // the process doesnt get one more (no remaining or i>= remainder)
            displs_send[i] = displs_send[i-1] + send_counts[i-1];
        }
        else {  //remainder = 0 and i = 0
            displs_send[i] =0;
        }
    }

    int count_body;
    count_body=1;    //how many kinds of data (block) my struct has
    MPI_Datatype array_of_types[count_body]; //count many elements
    array_of_types[0] = MPI_LONG_DOUBLE;
    int array_of_blocklengths[count_body];   // Says how many elements for each block
    array_of_blocklengths[0] = 5;
    
    MPI_Aint array_of_displaysments[count_body];    //says where every block starts in memory
    MPI_Aint address1, address2;
    MPI_Get_address(&bodies[0],&address1);
    MPI_Get_address(&bodies[0].mass,&address2);
    array_of_displaysments[0] = address2 - address1;
    
    /*Create MPI Datatype and commit*/
    MPI_Datatype body_type;
    MPI_Type_create_struct(count_body, array_of_blocklengths, array_of_displaysments, array_of_types, &body_type);
    MPI_Type_commit(&body_type);

    MPI_Scatterv(bodies, send_counts, displs_send, body_type, local_bodies, local_nBodies, body_type, 0, MPI_COMM_WORLD);

    //printf("The local num is: %d from %d\n", local_nBodies, self);
    //printf("The local body index 0 mass is: %Lf\n", local_bodies[0].mass);
    //printf("The nSteps is: %d\n", nSteps);


    for (int step = 0; step < nSteps; ++step) {
        //printf("Message: %s\n", "buraya geldik");
        if (imageStep > 0 && step % imageStep == 0)
            saveImage(step / imageStep, bodies, nBodies);   //do i need to change this part

        if (debug && self == 0)
            printf("%d\r", step);

        displs = malloc(np * sizeof(int));  //starting point for each process
        recv_counts = malloc(np * sizeof(int));
        recvbuf = malloc(nBodies * sizeof(body));
        
        /* Fill in the displacements and recv_counts */
        for (int i=0; i<np; i++) {
            recv_counts[i] = local_nBodies_woRemainder;
            //printf("the remainder is: %d\n", local_nBodies_woRemainder);
            if (i < remainder) {    //there is remainder and the process will have one remaining
                recv_counts[i]++;
                displs[i] = i * (local_nBodies_woRemainder +1);
            }
            else if (i != 0) {  // the process doesnt get one more (no remaining or i>= remainder)
                displs[i] = displs[i-1] + recv_counts[i-1];
            }
            else {  //remainder = 0 and i = 0
                displs[i] =0;
            }
            //printf("displs[1] %d\n", displs[1]);
            //if (i==4) {
            //    printf("The displs[self] is: %d when i is: %d from %d\n", displs[i], i, self);
            //}
            
        }
        //printf("the value of recv_counts is: %d from %d\n", recv_counts[self], self);
        //printf("The first body mass is: %Lf\n", local_bodies[0].mass);
        //if (self == 4) {
        //    printf("the value of the last element of process 4 is: %Lf\n", local_bodies[179].mass);
        //}
        MPI_Allgatherv( local_bodies, local_nBodies, body_type, recvbuf, recv_counts, displs, body_type, MPI_COMM_WORLD);
        //printf("the value of recvbuf last element is: %Lf\n", recvbuf[901].mass);

        if (TI->newton3) {
            // implementation with Newton's third law
            // used globally

            allToAll = (struct myBody**)malloc(nBodies * sizeof(struct myBody*));
            for (int i = 0; i < nBodies; ++i) {
                allToAll[i] = (struct myBody*)malloc(nBodies * sizeof(struct myBody));
                for (int j = 0; j < nBodies; ++j) {
                    allToAll[i][j].calculated=false;
                    allToAll[i][j].aX = 0.0;
                    allToAll[i][j].aY = 0.0;
                }
            }
            
            for (int i = 0; i < local_nBodies; ++i) {
                accelsX[i] = accelsY[i] = 0;
            }
            
            int count_mybody;
            count_mybody = 2;
            MPI_Datatype arrayOfTypes[count_mybody];
            arrayOfTypes[0] = MPI_BYTE;
            arrayOfTypes[1] = MPI_LONG_DOUBLE;
            // Says how many elements for block
            int arrayOfBlocklengths[count_mybody];
            arrayOfBlocklengths[0] = 1; //curly braces?
            arrayOfBlocklengths[1] = 2;
            
            MPI_Aint arrayOfDisplaysments[count_mybody];
            MPI_Aint myaddress1, myaddress2, myaddress3;
            MPI_Get_address(&allToAll[0][0],&myaddress1);
            MPI_Get_address(&allToAll[0][0].calculated,&myaddress2);
            MPI_Get_address(&allToAll[0][0].aX,&myaddress3);
            arrayOfDisplaysments[0] = myaddress2 - myaddress1;    //bool
            arrayOfDisplaysments[1] = myaddress3 - myaddress2;    //long double
            
            MPI_Datatype mybody_type;
            MPI_Type_create_struct(count_mybody, arrayOfBlocklengths, arrayOfDisplaysments, arrayOfTypes, &mybody_type);
            MPI_Type_commit(&mybody_type);
            
            int beginRow = displs[self];
            
            #pragma omp parallel for
            for (int i = beginRow; i < local_nBodies+beginRow; ++i) // local bodies
            {
                for (int j = 0; j < nBodies; ++j)   // all bodies
                {
                    int flag = 0;
                    MPI_Status status;
                    int sourceJ;
                    int order;
                    order = 0;
                    for (int k=0; k < np; k++) {
                        if (j < (order + recv_counts[k]) && j >= order) {
                            sourceJ = k;
                        }
                        order += recv_counts[k];
                    }
                    MPI_Iprobe(sourceJ, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
     
                    if (flag) {
                        // A matching message is available, so receive it
                        //printf(" i: %d j: %d self: %d\n", i, j, self);
                        MPI_Recv(&allToAll[j][i], 1, mybody_type, sourceJ, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        //printf("Message: %s\n", "buraya geldik");
                    }
                    
                    if ( allToAll[j][i].calculated == true ) {   //if it is calculated on the other one
                        //printf("Message: %s\n", "buraya geldikk");
                        accelsX[i-beginRow] -= allToAll[j][i].aX;
                        accelsY[i-beginRow] -= allToAll[j][i].aY;
                        //printf("Message: %s\n", "buraya geldik");
                    }
                    if (i == j)
                        continue;
                    else {
                        long double ax, ay;
                        compute(&recvbuf[i], &recvbuf[j], &ax, &ay);
                        accelsX[i-beginRow] += ax;
                        accelsY[i-beginRow] += ay;
                        
                        allToAll[i][j].calculated = true;
                        allToAll[i][j].aX = ax;
                        allToAll[i][j].aY = ay;
                        
                        int destinationJ;
                        int destOrder;
                        destOrder = 0;
                        for (int k=0; k < np; k++) {
                            if (j < (destOrder + recv_counts[k]) && j >= destOrder) {
                                destinationJ = k;
                                //printf("Message: %s\n", "buraya geldik");
                            }
                            destOrder += recv_counts[k];
                        }
                        MPI_Send(&allToAll[i][j], 1, mybody_type, destinationJ, 0, MPI_COMM_WORLD);
                        //printf("Message: %s\n", "buraya geldik");
                    }
                }
            }

        } else if (TI->newton3local) {
            // implementation with Newton's third law for
            // local computations

            iTOJ = (bool**)malloc(local_nBodies * sizeof(bool*));
            for (int i = 0; i < local_nBodies; ++i) {
                iTOJ[i] = (bool*)malloc(local_nBodies * sizeof(bool));
                for (int j = 0; j < local_nBodies; ++j) {
                    iTOJ[i][j] = false;
                }
            }

            for (int k = 0; k < local_nBodies; ++k) {
                accelsX[k] = accelsY[k] = 0;
            }

            #pragma omp parallel for
            for (int i = 0; i < local_nBodies; ++i) //local ones
            {
                for (int j = 0; j < nBodies; ++j) // should be all bodies
                {
                    int sourceOfJ;
                    int myOrder;
                    myOrder = 0;
                    for (int k=0; k < np; k++) {
                        if (j < (myOrder + recv_counts[k]) && j >= myOrder) {
                            sourceOfJ = k;
                        }
                        myOrder += recv_counts[k];
                    }
                    if (sourceOfJ == self) {    // j is local
                        int j_as_local = j-displs[self];
                        if (i != j_as_local && iTOJ[i][j_as_local]==false) {
                            long double ax, ay;
                            compute(&local_bodies[i],&recvbuf[j], &ax, &ay);
                            accelsX[i] += ax;
                            accelsY[i] += ay;
                            accelsX[j_as_local] -= ax;
                            accelsY[j_as_local] -= ay;

                            iTOJ[i][j_as_local]=true;
                            iTOJ[j_as_local][i]=true;
                        }
                    }
                    else {  //if j is not local
                        long double ax, ay;
                        compute(&local_bodies[i],&recvbuf[j], &ax, &ay);
                        accelsX[i] += ax;
                        accelsY[i] += ay;
                    }
                }
            }
    
        } else if (TI->approxSurrogate) {
            // implementation with big surroate bodies for bodies
            // in other processes (Master only)
        } else {
            // implementation without Newton's third law here
            // compute for each body duo
            for (int i = 0; i < local_nBodies; ++i)
            {
                accelsX[i] = accelsY[i] = 0;
                for (int j = 0; j < nBodies; ++j)
                {
                    int sourceOfJ;
                    int myOrder;
                    myOrder = 0;
                    for (int k=0; k < np; k++) {
                        if (j < (myOrder + recv_counts[k]) && j >= myOrder) {
                            sourceOfJ = k;
                        }
                        myOrder += recv_counts[k];
                    }
                    if (self == sourceOfJ && i == j-displs[sourceOfJ])
                        continue;
                    long double ax, ay;
                    compute(&local_bodies[i], &recvbuf[j], &ax, &ay);
                    accelsX[i] += ax;
                    accelsY[i] += ay;
                }
            }
        }
        //printf("Message: %s\n", "buraya geldik");
        // update every body
        #pragma omp parallel for
        for (int i = 0; i < local_nBodies; i++)
        {
            update(&local_bodies[i], deltaT, accelsX[i], accelsY[i]);
            //printf(" i: %d self: %d\n", i, self);
        }
    }   // end of steps
    
    finalBodies = malloc(nBodies * sizeof(body));
    // gather needed
    MPI_Gatherv(local_bodies, local_nBodies, body_type, finalBodies, recv_counts, displs, body_type, 0, MPI_COMM_WORLD);

    if (imageStep > 0 && nSteps%imageStep == 0)
        saveImage(nSteps/imageStep, finalBodies, nBodies);
    if (debug)
        printf("\n");

    //free(displs);
    //free(local_bodies);
    //free(recv_counts);
    //free(recvbuf);
    //free(allToAll);
    //free(iTOJ);
    //free(finalBodies);
}
