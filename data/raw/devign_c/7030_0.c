guint qemu_chr_fe_add_watch(CharDriverState *s, GIOCondition cond,

                            GIOFunc func, void *user_data)

{

    GSource *src;

    guint tag;



    if (s->chr_add_watch == NULL) {

        return -ENOSYS;

    }



    src = s->chr_add_watch(s, cond);

    g_source_set_callback(src, (GSourceFunc)func, user_data, NULL);

    tag = g_source_attach(src, NULL);

    g_source_unref(src);



    return tag;

}
