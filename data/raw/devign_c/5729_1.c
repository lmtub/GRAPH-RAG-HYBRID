static void park_frame_worker_threads(FrameThreadContext *fctx, int thread_count)

{

    int i;



    for (i = 0; i < thread_count; i++) {

        PerThreadContext *p = &fctx->threads[i];



        if (p->state != STATE_INPUT_READY) {

            pthread_mutex_lock(&p->progress_mutex);

            while (p->state != STATE_INPUT_READY)

                pthread_cond_wait(&p->output_cond, &p->progress_mutex);

            pthread_mutex_unlock(&p->progress_mutex);

        }


    }

}