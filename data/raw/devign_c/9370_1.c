static int vnc_worker_thread_loop(VncJobQueue *queue)

{

    VncJob *job;

    VncRectEntry *entry, *tmp;

    VncState vs;

    int n_rectangles;

    int saved_offset;



    vnc_lock_queue(queue);

    while (QTAILQ_EMPTY(&queue->jobs) && !queue->exit) {

        qemu_cond_wait(&queue->cond, &queue->mutex);

    }

    /* Here job can only be NULL if queue->exit is true */

    job = QTAILQ_FIRST(&queue->jobs);

    vnc_unlock_queue(queue);



    if (queue->exit) {

        return -1;

    }



    vnc_lock_output(job->vs);

    if (job->vs->csock == -1 || job->vs->abort == true) {

        vnc_unlock_output(job->vs);

        goto disconnected;

    }

    vnc_unlock_output(job->vs);



    /* Make a local copy of vs and switch output buffers */

    vnc_async_encoding_start(job->vs, &vs);



    /* Start sending rectangles */

    n_rectangles = 0;

    vnc_write_u8(&vs, VNC_MSG_SERVER_FRAMEBUFFER_UPDATE);

    vnc_write_u8(&vs, 0);

    saved_offset = vs.output.offset;

    vnc_write_u16(&vs, 0);



    vnc_lock_display(job->vs->vd);

    QLIST_FOREACH_SAFE(entry, &job->rectangles, next, tmp) {

        int n;



        if (job->vs->csock == -1) {

            vnc_unlock_display(job->vs->vd);



            goto disconnected;

        }



        n = vnc_send_framebuffer_update(&vs, entry->rect.x, entry->rect.y,

                                        entry->rect.w, entry->rect.h);



        if (n >= 0) {

            n_rectangles += n;

        }

        g_free(entry);

    }

    vnc_unlock_display(job->vs->vd);



    /* Put n_rectangles at the beginning of the message */

    vs.output.buffer[saved_offset] = (n_rectangles >> 8) & 0xFF;

    vs.output.buffer[saved_offset + 1] = n_rectangles & 0xFF;



    vnc_lock_output(job->vs);

    if (job->vs->csock != -1) {

        buffer_reserve(&job->vs->jobs_buffer, vs.output.offset);

        buffer_append(&job->vs->jobs_buffer, vs.output.buffer,

                      vs.output.offset);





	qemu_bh_schedule(job->vs->bh);

    }  else {



    }

    vnc_unlock_output(job->vs);



disconnected:

    vnc_lock_queue(queue);

    QTAILQ_REMOVE(&queue->jobs, job, next);

    vnc_unlock_queue(queue);

    qemu_cond_broadcast(&queue->cond);

    g_free(job);

    return 0;

}