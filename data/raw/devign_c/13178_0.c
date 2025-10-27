static int target_to_host_fcntl_cmd(int cmd)

{

    switch(cmd) {

	case TARGET_F_DUPFD:

	case TARGET_F_GETFD:

	case TARGET_F_SETFD:

	case TARGET_F_GETFL:

	case TARGET_F_SETFL:

            return cmd;

        case TARGET_F_GETLK:

	    return F_GETLK;

	case TARGET_F_SETLK:

	    return F_SETLK;

	case TARGET_F_SETLKW:

	    return F_SETLKW;

	case TARGET_F_GETOWN:

	    return F_GETOWN;

	case TARGET_F_SETOWN:

	    return F_SETOWN;

	case TARGET_F_GETSIG:

	    return F_GETSIG;

	case TARGET_F_SETSIG:

	    return F_SETSIG;

#if TARGET_ABI_BITS == 32

        case TARGET_F_GETLK64:

	    return F_GETLK64;

	case TARGET_F_SETLK64:

	    return F_SETLK64;

	case TARGET_F_SETLKW64:

	    return F_SETLKW64;

#endif

        case TARGET_F_SETLEASE:

            return F_SETLEASE;

        case TARGET_F_GETLEASE:

            return F_GETLEASE;

#ifdef F_DUPFD_CLOEXEC

        case TARGET_F_DUPFD_CLOEXEC:

            return F_DUPFD_CLOEXEC;

#endif

        case TARGET_F_NOTIFY:

            return F_NOTIFY;

#ifdef F_GETOWN_EX

	case TARGET_F_GETOWN_EX:

	    return F_GETOWN_EX;

#endif

#ifdef F_SETOWN_EX

	case TARGET_F_SETOWN_EX:

	    return F_SETOWN_EX;

#endif

	default:

            return -TARGET_EINVAL;

    }

    return -TARGET_EINVAL;

}
