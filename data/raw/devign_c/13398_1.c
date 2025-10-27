int qemu_socket(int domain, int type, int protocol)

{

    int ret;



#ifdef SOCK_CLOEXEC

    ret = socket(domain, type | SOCK_CLOEXEC, protocol);

#else

    ret = socket(domain, type, protocol);

    if (ret >= 0) {

        qemu_set_cloexec(ret);

    }

#endif



    return ret;

}
