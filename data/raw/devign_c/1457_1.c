static void fd_chr_update_read_handler(CharDriverState *chr)

{

    FDCharDriver *s = chr->opaque;



    if (s->fd_in_tag) {

        g_source_remove(s->fd_in_tag);

        s->fd_in_tag = 0;

    }



    if (s->fd_in) {

        s->fd_in_tag = io_add_watch_poll(s->fd_in, fd_chr_read_poll, fd_chr_read, chr);

    }

}
