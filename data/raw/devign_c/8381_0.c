static inline abi_long do_msgsnd(int msqid, abi_long msgp,

                                 unsigned int msgsz, int msgflg)

{

    struct target_msgbuf *target_mb;

    struct msgbuf *host_mb;

    abi_long ret = 0;



    if (!lock_user_struct(VERIFY_READ, target_mb, msgp, 0))

        return -TARGET_EFAULT;

    host_mb = malloc(msgsz+sizeof(long));

    host_mb->mtype = (abi_long) tswapal(target_mb->mtype);

    memcpy(host_mb->mtext, target_mb->mtext, msgsz);

    ret = get_errno(msgsnd(msqid, host_mb, msgsz, msgflg));

    free(host_mb);

    unlock_user_struct(target_mb, msgp, 0);



    return ret;

}
