void ff_thread_report_progress(ThreadFrame *f, int n, int field)

{

    PerThreadContext *p;

    atomic_int *progress = f->progress ? (atomic_int*)f->progress->data : NULL;



    if (!progress ||

        atomic_load_explicit(&progress[field], memory_order_relaxed) >= n)

        return;



    p = f->owner[field]->internal->thread_ctx;



    if (f->owner[field]->debug&FF_DEBUG_THREADS)

        av_log(f->owner[field], AV_LOG_DEBUG,

               "%p finished %d field %d\n", progress, n, field);



    pthread_mutex_lock(&p->progress_mutex);



    atomic_store_explicit(&progress[field], n, memory_order_release);



    pthread_cond_broadcast(&p->progress_cond);

    pthread_mutex_unlock(&p->progress_mutex);

}
