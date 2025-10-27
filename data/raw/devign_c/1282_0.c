CharDriverState *qemu_chr_open_eventfd(int eventfd)

{

    CharDriverState *chr = qemu_chr_open_fd(eventfd, eventfd);



    if (chr) {

        chr->avail_connections = 1;

    }



    return chr;

}
