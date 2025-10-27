static int nbd_can_accept(void)

{

    return nb_fds < shared;

}
