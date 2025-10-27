static int nbd_errno_to_system_errno(int err)

{

    switch (err) {

    case NBD_SUCCESS:

        return 0;

    case NBD_EPERM:

        return EPERM;

    case NBD_EIO:

        return EIO;

    case NBD_ENOMEM:

        return ENOMEM;

    case NBD_ENOSPC:

        return ENOSPC;

    case NBD_EINVAL:

    default:

        return EINVAL;

    }

}
