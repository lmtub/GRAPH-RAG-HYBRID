static int target_pread(int fd, abi_ulong ptr, abi_ulong len,

                        abi_ulong offset)

{

    void *buf;

    int ret;



    buf = lock_user(VERIFY_WRITE, ptr, len, 0);




    ret = pread(fd, buf, len, offset);




    unlock_user(buf, ptr, len);

    return ret;
