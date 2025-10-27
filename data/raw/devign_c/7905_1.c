static av_always_inline void thread_park_workers(SliceThreadContext *c, int thread_count)

{

    while (c->current_job != thread_count + c->job_count)

        pthread_cond_wait(&c->last_job_cond, &c->current_job_lock);

    pthread_mutex_unlock(&c->current_job_lock);

}
