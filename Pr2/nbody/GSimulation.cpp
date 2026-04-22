/*
    This file is part of the example codes which have been used
    for the "Code Optmization Workshop".
    
    Copyright (C) 2016  Fabio Baruffa <fbaru-dev@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "GSimulation.hpp"
#include "cpu_time.hpp"

#include <sycl/sycl.hpp>


#define TILE_SIZE 256


using  namespace  sycl;


GSimulation :: GSimulation()
{
  std::cout << "===============================" << std::endl;
  std::cout << " Initialize Gravity Simulation" << std::endl;
  set_npart(16000); 
  set_nsteps(10);
  set_tstep(0.1); 
  set_sfreq(1);
}

void GSimulation :: set_number_of_particles(int N)  
{
  set_npart(N);
}

void GSimulation :: set_number_of_steps(int N)  
{
  set_nsteps(N);
}

void GSimulation :: init_pos()  
{
  std::random_device rd;	//random number generator
  std::mt19937 gen(42);      
  std::uniform_real_distribution<real_type> unif_d(0,1.0);
  
  for(int i=0; i<get_npart(); ++i)
  {
    particles[i].pos[0] = unif_d(gen);
    particles[i].pos[1] = unif_d(gen);
    particles[i].pos[2] = unif_d(gen);
  }
}

void GSimulation :: init_vel()  
{
  std::random_device rd;        //random number generator
  std::mt19937 gen(42);
  std::uniform_real_distribution<real_type> unif_d(-1.0,1.0);

  for(int i=0; i<get_npart(); ++i)
  {
    particles[i].vel[0] = unif_d(gen) * 1.0e-3f;
    particles[i].vel[1] = unif_d(gen) * 1.0e-3f;
    particles[i].vel[2] = unif_d(gen) * 1.0e-3f; 
  }
}

void GSimulation :: init_acc() 
{
  for(int i=0; i<get_npart(); ++i)
  {
    particles[i].acc[0] = 0.f; 
    particles[i].acc[1] = 0.f;
    particles[i].acc[2] = 0.f;
  }
}

void GSimulation :: init_mass() 
{
  real_type n   = static_cast<real_type> (get_npart());
  std::random_device rd;        //random number generator
  std::mt19937 gen(42);
  std::uniform_real_distribution<real_type> unif_d(0.0,1.0);

  for(int i=0; i<get_npart(); ++i)
  {
    particles[i].mass = n * unif_d(gen); 
  }
}

void GSimulation :: get_acceleration(int n)
{
   int i,j;

   const float softeningSquared = 1e-3f;
   const float G = 6.67259e-11f;

   for (i = 0; i < n; i++)// update acceleration
   {
     real_type ax_i = particles[i].acc[0];
     real_type ay_i = particles[i].acc[1];
     real_type az_i = particles[i].acc[2];
     for (j = 0; j < n; j++)
     {
         real_type dx, dy, dz;
	 real_type distanceSqr = 0.0f;
	 real_type distanceInv = 0.0f;
		  
	 dx = particles[j].pos[0] - particles[i].pos[0];	//1flop
	 dy = particles[j].pos[1] - particles[i].pos[1];	//1flop	
	 dz = particles[j].pos[2] - particles[i].pos[2];	//1flop
	
 	 distanceSqr = dx*dx + dy*dy + dz*dz + softeningSquared;	//6flops
 	 distanceInv = 1.0f / sqrtf(distanceSqr);			//1div+1sqrt

	 ax_i += dx * G * particles[j].mass * distanceInv * distanceInv * distanceInv; //6flops
	 ay_i += dy * G * particles[j].mass * distanceInv * distanceInv * distanceInv; //6flops
	 az_i += dz * G * particles[j].mass * distanceInv * distanceInv * distanceInv; //6flops
     }
     particles[i].acc[0] = ax_i;
     particles[i].acc[1] = ay_i;
     particles[i].acc[2] = az_i;
   }
}



void GSimulation :: get_acceleration_gpu(sycl::queue Q, int n)
{
   int i,j;

   const float softeningSquared = 1e-3f;
   const float G = 6.67259e-11f;

   auto pos_x = nparticles.pos_x;
   auto pos_y = nparticles.pos_y;
   auto pos_z = nparticles.pos_z;

   auto acc_x = nparticles.acc_x;
   auto acc_y = nparticles.acc_y;
   auto acc_z = nparticles.acc_z;

   auto mass = nparticles.mass;

   
    Q.submit([&](handler & h){
      range global = range <1> (n);
      range local = range <1> (TILE_SIZE);

      local_accessor<real_type, 1> tile_pos_x(range<1> (TILE_SIZE), h);
      local_accessor<real_type, 1> tile_pos_y(range<1> (TILE_SIZE), h);
      local_accessor<real_type, 1> tile_pos_z(range<1> (TILE_SIZE), h);

      local_accessor<real_type, 1> tile_mass(range<1> (TILE_SIZE), h);

      h.parallel_for(nd_range<1> (global, local), [=](nd_item <1> item){
        int global_idx = item.get_global_id(0);

        int local_idx = item.get_local_id(0);

        real_type xi = pos_x[global_idx];
        real_type yi = pos_y[global_idx];
        real_type zi = pos_z[global_idx];

        real_type ax = acc_x[global_idx];
        real_type ay = acc_y[global_idx];
        real_type az = acc_z[global_idx];

        real_type distanceSqr = 0.0f;
	      real_type distanceInv = 0.0f;
        real_type dx, dy, dz;

        for (int j = 0; j < n; j += TILE_SIZE){ //Suponemos que n es multiplo de TILE_SIZE
          int global_j = j + local_idx;

          tile_pos_x[local_idx] = pos_x[global_j];
          tile_pos_y[local_idx] = pos_y[global_j];
          tile_pos_z[local_idx] = pos_z[global_j];

          tile_mass[local_idx] = mass[global_j];

          item.barrier();

          for (int jj = 0; jj < TILE_SIZE; ++jj){

            dx = tile_pos_x[jj] - xi;	//1flop
	          dy = tile_pos_y[jj] - yi;	//1flop	
	          dz = tile_pos_z[jj] - zi;

            distanceSqr = dx*dx + dy*dy + dz*dz + softeningSquared;	//6flops
 	          distanceInv = 1.0f / sqrtf(distanceSqr);			//1div+1sqrt
            ax += dx * G * tile_mass[jj] * distanceInv * distanceInv * distanceInv; //6flops
	          ay += dy * G * tile_mass[jj] * distanceInv * distanceInv * distanceInv; //6flops
	          az += dz * G * tile_mass[jj] * distanceInv * distanceInv * distanceInv; //6flops
          }
          item.barrier();

          
        }

        acc_x[global_idx] = ax;
        acc_y[global_idx] = ay;
        acc_z[global_idx] = az;
        
      });

    }).wait();
}

real_type GSimulation :: updateParticles(int n, real_type dt)
{
   int i;
   real_type energy = 0;

   for (i = 0; i < n; ++i)// update position
   {
     particles[i].vel[0] += particles[i].acc[0] * dt; //2flops
     particles[i].vel[1] += particles[i].acc[1] * dt; //2flops
     particles[i].vel[2] += particles[i].acc[2] * dt; //2flops
	  
     particles[i].pos[0] += particles[i].vel[0] * dt; //2flops
     particles[i].pos[1] += particles[i].vel[1] * dt; //2flops
     particles[i].pos[2] += particles[i].vel[2] * dt; //2flops

     particles[i].acc[0] = 0.;
     particles[i].acc[1] = 0.;
     particles[i].acc[2] = 0.;
	
     energy += particles[i].mass * (
	      particles[i].vel[0]*particles[i].vel[0] + 
               particles[i].vel[1]*particles[i].vel[1] +
               particles[i].vel[2]*particles[i].vel[2]); //7flops
   }
   return energy;
}

real_type GSimulation :: updateParticles_gpu(sycl::queue Q, int n, real_type dt)
{
   int i;
   real_type energy = 0;
   
   real_type *tmp_energy = malloc_shared<real_type>(1, Q);
   *tmp_energy = 0;

   auto vel_x = nparticles.vel_x;
   auto vel_y = nparticles.vel_y;
   auto vel_z = nparticles.vel_z;

   auto acc_x = nparticles.acc_x;
   auto acc_y = nparticles.acc_y;
   auto acc_z = nparticles.acc_z;

   auto pos_x = nparticles.pos_x;
   auto pos_y = nparticles.pos_y;
   auto pos_z = nparticles.pos_z;

   auto mass = nparticles.mass;


   Q.submit([&](handler & h){
    h.parallel_for(n, [=](id<1> id){
      int i = id[0];

      real_type m = mass[i];
      real_type vx = vel_x[i]; 
      real_type vy = vel_y[i];
      real_type vz = vel_z[i];

      real_type ax = acc_x[i];
      real_type ay = acc_y[i];
      real_type az = acc_z[i];

      real_type px = pos_x[i];
      real_type py = pos_y[i];
      real_type pz = pos_z[i];

      vx += ax*dt;
      vy += ay*dt;
      vz += az*dt;

      px += vx*dt;
      py += vy*dt;
      pz += vz*dt;
      
      real_type value = m*(vx*vx + vy*vy + vz*vz);

      auto v = sycl::atomic_ref <real_type, sycl::memory_order::acq_rel,
                                  sycl::memory_scope::device,
                                  sycl::access::address_space::global_space> (*tmp_energy);

      v.fetch_add(value);

      acc_x[i] = 0.0;
      acc_y[i] = 0.0;
      acc_z[i] = 0.0;

      vel_x[i] = vx;
      vel_y[i] = vy;
      vel_z[i] = vz;

      pos_x[i] = px;
      pos_y[i] = py;
      pos_z[i] = py;

    });


   }).wait(); 


   energy += *tmp_energy;

   free(tmp_energy, Q);

   return energy;
}






void GSimulation :: start() 
{
  real_type energy;
  real_type dt = get_tstep();
  int n = get_npart();

  //allocate particles
  particles = new ParticleAoS[n];

  init_pos();
  init_vel();
  init_acc();
  init_mass();
  
  print_header();
  
  _totTime = 0.; 
  
  
  CPUTime time;
  double ts0 = 0;
  double ts1 = 0;
  double nd = double(n);
  double gflops = 1e-9 * ( (11. + 18. ) * nd*nd  +  nd * 19. );
  double av=0.0, dev=0.0;
  int nf = 0;


  //Inicializo las variables para ejecucion en SYCL 
  sycl::queue Q(sycl::gpu_selector_v); //Creo la cola de ejecución

  nparticles = ParticleSoA();

  nparticles.pos_x = malloc_shared<real_type>(n, Q);
  nparticles.pos_y = malloc_shared<real_type>(n, Q);
  nparticles.pos_z = malloc_shared<real_type>(n, Q);

  nparticles.vel_x = malloc_shared<real_type>(n, Q);
  nparticles.vel_y = malloc_shared<real_type>(n, Q);
  nparticles.vel_z = malloc_shared<real_type>(n, Q);

  nparticles.acc_x = malloc_shared<real_type>(n, Q);
  nparticles.acc_y = malloc_shared<real_type>(n, Q);
  nparticles.acc_z = malloc_shared<real_type>(n, Q);

  nparticles.mass = malloc_shared<real_type>(n, Q);

  for (int i = 0; i < n; ++i){
    nparticles.pos_x[i] = particles[i].pos[0];
    nparticles.pos_y[i] = particles[i].pos[1];
    nparticles.pos_z[i] = particles[i].pos[2];

    nparticles.vel_x[i] = particles[i].vel[0];
    nparticles.vel_y[i] = particles[i].vel[1];
    nparticles.vel_z[i] = particles[i].vel[2];

    nparticles.acc_x[i] = particles[i].acc[0]; 
    nparticles.acc_y[i] = particles[i].acc[1]; 
    nparticles.acc_z[i] = particles[i].acc[2]; 

    nparticles.mass[i] = particles[i].mass;
  }
  


  
  const double t0 = time.start();
  for (int s=1; s<=get_nsteps(); ++s)
  {   
   ts0 += time.start(); 


    //get_acceleration(n);

    get_acceleration_gpu(Q, n);

    //energy = updateParticles(n, dt);

    energy = updateParticles_gpu(Q, n, dt);
    _kenergy = 0.5 * energy; 
    
    ts1 += time.stop();
    if(!(s%get_sfreq()) ) 
    {
      nf += 1;      
      std::cout << " " 
		<<  std::left << std::setw(8)  << s
		<<  std::left << std::setprecision(5) << std::setw(8)  << s*get_tstep()
		<<  std::left << std::setprecision(5) << std::setw(12) << _kenergy
		<<  std::left << std::setprecision(5) << std::setw(12) << (ts1 - ts0)
		<<  std::left << std::setprecision(5) << std::setw(12) << gflops*get_sfreq()/(ts1 - ts0)
		<<  std::endl;
      if(nf > 2) 
      {
	av  += gflops*get_sfreq()/(ts1 - ts0);
	dev += gflops*get_sfreq()*gflops*get_sfreq()/((ts1-ts0)*(ts1-ts0));
      }
      
      ts0 = 0;
      ts1 = 0;
    }
  
  } //end of the time step loop
  
  const double t1 = time.stop();
  _totTime  = (t1-t0);
  _totFlops = gflops*get_nsteps();
  
  av/=(double)(nf-2);
  dev= std::sqrt(dev/(double)(nf-2)-av*av);
  

  std::cout << std::endl;
  std::cout << "# Total Time (s)      : " << _totTime << std::endl;
  std::cout << "# Average Performance : " << av << " +- " <<  dev << std::endl;
  std::cout << "===============================" << std::endl;


  free(nparticles.pos_x, Q);
  free(nparticles.pos_y, Q);
  free(nparticles.pos_z, Q);
  free(nparticles.acc_x, Q);
  free(nparticles.acc_y, Q);
  free(nparticles.acc_z, Q);
  free(nparticles.vel_x, Q);
  free(nparticles.vel_y, Q);
  free(nparticles.vel_z, Q);
  free(nparticles.mass, Q);
}


void GSimulation :: print_header()
{
	    
  std::cout << " nPart = " << get_npart()  << "; " 
	    << "nSteps = " << get_nsteps() << "; " 
	    << "dt = "     << get_tstep()  << std::endl;
	    
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << " " 
	    <<  std::left << std::setw(8)  << "s"
	    <<  std::left << std::setw(8)  << "dt"
	    <<  std::left << std::setw(12) << "kenergy"
	    <<  std::left << std::setw(12) << "time (s)"
	    <<  std::left << std::setw(12) << "GFlops"
	    <<  std::endl;
  std::cout << "------------------------------------------------" << std::endl;


}

GSimulation :: ~GSimulation()
{
  delete particles;
}
