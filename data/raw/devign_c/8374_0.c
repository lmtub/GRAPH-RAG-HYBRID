static void rng_egd_finalize(Object *obj)

{

    RngEgd *s = RNG_EGD(obj);



    if (s->chr) {

        qemu_chr_add_handlers(s->chr, NULL, NULL, NULL, NULL);

        qemu_chr_fe_release(s->chr);

    }



    g_free(s->chr_name);



    rng_egd_free_requests(s);

}
