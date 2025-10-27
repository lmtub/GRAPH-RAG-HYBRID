void stream_start(const char *job_id, BlockDriverState *bs,

                  BlockDriverState *base, const char *backing_file_str,

                  int64_t speed, BlockdevOnError on_error,

                  BlockCompletionFunc *cb, void *opaque, Error **errp)

{

    StreamBlockJob *s;



    s = block_job_create(job_id, &stream_job_driver, bs, speed,

                         cb, opaque, errp);

    if (!s) {

        return;

    }



    s->base = base;

    s->backing_file_str = g_strdup(backing_file_str);



    s->on_error = on_error;

    s->common.co = qemu_coroutine_create(stream_run);

    trace_stream_start(bs, base, s, s->common.co, opaque);

    qemu_coroutine_enter(s->common.co, s);

}
