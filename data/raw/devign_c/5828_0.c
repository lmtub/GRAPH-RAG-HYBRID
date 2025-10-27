static int nbd_can_accept(void *opaque)

{

    return nb_fds < shared;

}
