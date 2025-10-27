static void coroutine_fn mirror_pause(BlockJob *job)

{

    MirrorBlockJob *s = container_of(job, MirrorBlockJob, common);



    mirror_drain(s);

}
