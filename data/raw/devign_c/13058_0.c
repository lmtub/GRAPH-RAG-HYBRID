static abi_long lock_iovec(int type, struct iovec *vec, abi_ulong target_addr,

                           int count, int copy)

{

    struct target_iovec *target_vec;

    abi_ulong base;

    int i;



    target_vec = lock_user(VERIFY_READ, target_addr, count * sizeof(struct target_iovec), 1);

    if (!target_vec)

        return -TARGET_EFAULT;

    for(i = 0;i < count; i++) {

        base = tswapal(target_vec[i].iov_base);

        vec[i].iov_len = tswapal(target_vec[i].iov_len);

        if (vec[i].iov_len != 0) {

            vec[i].iov_base = lock_user(type, base, vec[i].iov_len, copy);

            /* Don't check lock_user return value. We must call writev even

               if a element has invalid base address. */

        } else {

            /* zero length pointer is ignored */

            vec[i].iov_base = NULL;

        }

    }

    unlock_user (target_vec, target_addr, 0);

    return 0;

}
