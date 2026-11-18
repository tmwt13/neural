#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

pthread_cond_t work_done;
pthread_cond_t work_start;
pthread_mutex_t mutex;
int counter = 0;

void *worker(void* args){
  int id = *(int*)args;
  printf("THREAD\n");
  for(;;) {
    pthread_mutex_lock(&mutex);
    if (counter == 0) {
      pthread_cond_broadcast(&work_done);
      pthread_cond_wait(&work_start,&mutex);
      printf("NEW WORK on %d\n",id);
      pthread_mutex_unlock(&mutex);
      continue;
    }
    counter--;
    pthread_mutex_unlock(&mutex);
    printf("work in %d\n", id);
    sleep(2);
  }
  printf("TERMINATED\n");
}


int main(){
  pthread_mutex_init(&mutex, NULL);
  pthread_cond_init(&work_done, NULL);
  pthread_cond_init(&work_start, NULL);
  pthread_t t1; int id1 = 1;
  pthread_create(&t1, NULL, worker, &id1);
  pthread_t t2; int id2 = 2;
  pthread_create(&t2, NULL, worker, &id2);

  sleep(2);
  printf("start\n");
  pthread_mutex_lock(&mutex);
  counter = 3;
  printf("counter == 3\n");
  pthread_cond_broadcast(&work_start);
  sleep(1);
  pthread_mutex_unlock(&mutex);
  printf("unlocked\n");

  pthread_mutex_lock(&mutex);
  pthread_cond_wait(&work_done, &mutex);
  pthread_mutex_unlock(&mutex);

  printf("DONE...\n");
  

  return 0;
}
