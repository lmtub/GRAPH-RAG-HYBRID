static inline abi_long do_msgrcv(int msqid, abi_long msgp,

                                 unsigned int msgsz, abi_long msgtyp,

                                 int msgflg)

{

    struct target_msgbuf *target_mb;

    char *target_mtext;

    struct msgbuf *host_mb;

    abi_long ret = 0;



    if (!lock_user_struct(VERIFY_WRITE, target_mb, msgp, 0))

        return -TARGET_EFAULT;



    host_mb = malloc(msgsz+sizeof(long));

    ret = get_errno(msgrcv(msqid, host_mb, msgsz, tswapal(msgtyp), msgflg));



    if (ret > 0) {

        abi_ulong target_mtext_addr = msgp + sizeof(abi_ulong);

        target_mtext = lock_user(VERIFY_WRITE, target_mtext_addr, ret, 0);

        if (!target_mtext) {

            ret = -TARGET_EFAULT;

            goto end;

        }

        memcpy(target_mb->mtext, host_mb->mtext, ret);

        unlock_user(target_mtext, target_mtext_addr, ret);

    }



    target_mb->mtype = tswapal(host_mb->mtype);

    free(host_mb);



end:

    if (target_mb)

        unlock_user_struct(target_mb, msgp, 1);

    return ret;

}
