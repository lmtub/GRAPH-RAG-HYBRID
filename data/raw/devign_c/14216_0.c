static void rng_random_finalize(Object *obj)

{

    RndRandom *s = RNG_RANDOM(obj);



    qemu_set_fd_handler(s->fd, NULL, NULL, NULL);



    if (s->fd != -1) {

        qemu_close(s->fd);

    }



    g_free(s->filename);

}
