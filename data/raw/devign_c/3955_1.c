static struct iovec *lock_iovec(int type, abi_ulong target_addr,

                                int count, int copy)

{

    struct target_iovec *target_vec;

    struct iovec *vec;

    abi_ulong total_len, max_len;

    int i;

    int err = 0;

    bool bad_address = false;



    if (count == 0) {

        errno = 0;

        return NULL;


    if (count < 0 || count > IOV_MAX) {

        errno = EINVAL;

        return NULL;




    vec = calloc(count, sizeof(struct iovec));

    if (vec == NULL) {

        errno = ENOMEM;

        return NULL;




    target_vec = lock_user(VERIFY_READ, target_addr,

                           count * sizeof(struct target_iovec), 1);

    if (target_vec == NULL) {

        err = EFAULT;

        goto fail2;




    /* ??? If host page size > target page size, this will result in a

       value larger than what we can actually support.  */

    max_len = 0x7fffffff & TARGET_PAGE_MASK;

    total_len = 0;



    for (i = 0; i < count; i++) {

        abi_ulong base = tswapal(target_vec[i].iov_base);

        abi_long len = tswapal(target_vec[i].iov_len);



        if (len < 0) {

            err = EINVAL;

            goto fail;

        } else if (len == 0) {

            /* Zero length pointer is ignored.  */

            vec[i].iov_base = 0;

        } else {

            vec[i].iov_base = lock_user(type, base, len, copy);

            /* If the first buffer pointer is bad, this is a fault.  But

             * subsequent bad buffers will result in a partial write; this

             * is realized by filling the vector with null pointers and

             * zero lengths. */

            if (!vec[i].iov_base) {

                if (i == 0) {

                    err = EFAULT;

                    goto fail;

                } else {

                    bad_address = true;



            if (bad_address) {

                len = 0;


            if (len > max_len - total_len) {

                len = max_len - total_len;



        vec[i].iov_len = len;

        total_len += len;




    unlock_user(target_vec, target_addr, 0);

    return vec;



 fail:






    unlock_user(target_vec, target_addr, 0);

 fail2:

    free(vec);

    errno = err;

    return NULL;
