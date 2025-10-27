static abi_long do_ipc(unsigned int call, abi_long first,

                       abi_long second, abi_long third,

                       abi_long ptr, abi_long fifth)

{

    int version;

    abi_long ret = 0;



    version = call >> 16;

    call &= 0xffff;



    switch (call) {

    case IPCOP_semop:

        ret = do_semop(first, ptr, second);

        break;



    case IPCOP_semget:

        ret = get_errno(semget(first, second, third));

        break;



    case IPCOP_semctl: {

        /* The semun argument to semctl is passed by value, so dereference the

         * ptr argument. */

        abi_ulong atptr;

        get_user_ual(atptr, ptr);

        ret = do_semctl(first, second, third, atptr);

        break;

    }



    case IPCOP_msgget:

        ret = get_errno(msgget(first, second));

        break;



    case IPCOP_msgsnd:

        ret = do_msgsnd(first, ptr, second, third);

        break;



    case IPCOP_msgctl:

        ret = do_msgctl(first, second, ptr);

        break;



    case IPCOP_msgrcv:

        switch (version) {

        case 0:

            {

                struct target_ipc_kludge {

                    abi_long msgp;

                    abi_long msgtyp;

                } *tmp;



                if (!lock_user_struct(VERIFY_READ, tmp, ptr, 1)) {

                    ret = -TARGET_EFAULT;

                    break;

                }



                ret = do_msgrcv(first, tswapal(tmp->msgp), second, tswapal(tmp->msgtyp), third);



                unlock_user_struct(tmp, ptr, 0);

                break;

            }

        default:

            ret = do_msgrcv(first, ptr, second, fifth, third);

        }

        break;



    case IPCOP_shmat:

        switch (version) {

        default:

        {

            abi_ulong raddr;

            raddr = do_shmat(first, ptr, second);

            if (is_error(raddr))

                return get_errno(raddr);

            if (put_user_ual(raddr, third))

                return -TARGET_EFAULT;

            break;

        }

        case 1:

            ret = -TARGET_EINVAL;

            break;

        }

	break;

    case IPCOP_shmdt:

        ret = do_shmdt(ptr);

	break;



    case IPCOP_shmget:

	/* IPC_* flag values are the same on all linux platforms */

	ret = get_errno(shmget(first, second, third));

	break;



	/* IPC_* and SHM_* command values are the same on all linux platforms */

    case IPCOP_shmctl:

        ret = do_shmctl(first, second, ptr);

        break;

    default:

	gemu_log("Unsupported ipc call: %d (version %d)\n", call, version);

	ret = -TARGET_ENOSYS;

	break;

    }

    return ret;

}
