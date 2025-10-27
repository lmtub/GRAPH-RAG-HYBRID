static inline abi_long do_semctl(int semid, int semnum, int cmd,

                                 union target_semun target_su)

{

    union semun arg;

    struct semid_ds dsarg;

    unsigned short *array;

    struct seminfo seminfo;

    abi_long ret = -TARGET_EINVAL;

    abi_long err;

    cmd &= 0xff;



    switch( cmd ) {

	case GETVAL:

	case SETVAL:

            arg.val = tswapl(target_su.val);

            ret = get_errno(semctl(semid, semnum, cmd, arg));

            target_su.val = tswapl(arg.val);

            break;

	case GETALL:

	case SETALL:

            err = target_to_host_semarray(semid, &array, target_su.array);

            if (err)

                return err;

            arg.array = array;

            ret = get_errno(semctl(semid, semnum, cmd, arg));

            err = host_to_target_semarray(semid, target_su.array, &array);

            if (err)

                return err;

            break;

	case IPC_STAT:

	case IPC_SET:

	case SEM_STAT:

            err = target_to_host_semid_ds(&dsarg, target_su.buf);

            if (err)

                return err;

            arg.buf = &dsarg;

            ret = get_errno(semctl(semid, semnum, cmd, arg));

            err = host_to_target_semid_ds(target_su.buf, &dsarg);

            if (err)

                return err;

            break;

	case IPC_INFO:

	case SEM_INFO:

            arg.__buf = &seminfo;

            ret = get_errno(semctl(semid, semnum, cmd, arg));

            err = host_to_target_seminfo(target_su.__buf, &seminfo);

            if (err)

                return err;

            break;

	case IPC_RMID:

	case GETPID:

	case GETNCNT:

	case GETZCNT:

            ret = get_errno(semctl(semid, semnum, cmd, NULL));

            break;

    }



    return ret;

}
