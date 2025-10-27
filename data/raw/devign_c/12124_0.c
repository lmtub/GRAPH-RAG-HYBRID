static int slirp_state_load(QEMUFile *f, void *opaque, int version_id)

{

    Slirp *slirp = opaque;

    struct ex_list *ex_ptr;



    while (qemu_get_byte(f)) {

        int ret;

        struct socket *so = socreate(slirp);



        if (!so)

            return -ENOMEM;



        ret = vmstate_load_state(f, &vmstate_slirp_socket, so, version_id);



        if (ret < 0)

            return ret;



        if ((so->so_faddr.s_addr & slirp->vnetwork_mask.s_addr) !=

            slirp->vnetwork_addr.s_addr) {

            return -EINVAL;

        }

        for (ex_ptr = slirp->exec_list; ex_ptr; ex_ptr = ex_ptr->ex_next) {

            if (ex_ptr->ex_pty == 3 &&

                so->so_faddr.s_addr == ex_ptr->ex_addr.s_addr &&

                so->so_fport == ex_ptr->ex_fport) {

                break;

            }

        }

        if (!ex_ptr)

            return -EINVAL;



        so->extra = (void *)ex_ptr->ex_exec;

    }



    if (version_id >= 2) {

        slirp->ip_id = qemu_get_be16(f);

    }



    if (version_id >= 3) {

        slirp_bootp_load(f, slirp);

    }



    return 0;

}
