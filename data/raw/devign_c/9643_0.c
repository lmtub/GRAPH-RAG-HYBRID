static inline abi_long target_to_host_cmsg(struct msghdr *msgh,

                                           struct target_msghdr *target_msgh)

{

    struct cmsghdr *cmsg = CMSG_FIRSTHDR(msgh);

    abi_long msg_controllen;

    abi_ulong target_cmsg_addr;

    struct target_cmsghdr *target_cmsg;

    socklen_t space = 0;

    

    msg_controllen = tswapal(target_msgh->msg_controllen);

    if (msg_controllen < sizeof (struct target_cmsghdr)) 

        goto the_end;

    target_cmsg_addr = tswapal(target_msgh->msg_control);

    target_cmsg = lock_user(VERIFY_READ, target_cmsg_addr, msg_controllen, 1);

    if (!target_cmsg)

        return -TARGET_EFAULT;



    while (cmsg && target_cmsg) {

        void *data = CMSG_DATA(cmsg);

        void *target_data = TARGET_CMSG_DATA(target_cmsg);



        int len = tswapal(target_cmsg->cmsg_len)

                  - TARGET_CMSG_ALIGN(sizeof (struct target_cmsghdr));



        space += CMSG_SPACE(len);

        if (space > msgh->msg_controllen) {

            space -= CMSG_SPACE(len);

            gemu_log("Host cmsg overflow\n");

            break;

        }



        if (tswap32(target_cmsg->cmsg_level) == TARGET_SOL_SOCKET) {

            cmsg->cmsg_level = SOL_SOCKET;

        } else {

            cmsg->cmsg_level = tswap32(target_cmsg->cmsg_level);

        }

        cmsg->cmsg_type = tswap32(target_cmsg->cmsg_type);

        cmsg->cmsg_len = CMSG_LEN(len);



        if (cmsg->cmsg_level != SOL_SOCKET || cmsg->cmsg_type != SCM_RIGHTS) {

            gemu_log("Unsupported ancillary data: %d/%d\n", cmsg->cmsg_level, cmsg->cmsg_type);

            memcpy(data, target_data, len);

        } else {

            int *fd = (int *)data;

            int *target_fd = (int *)target_data;

            int i, numfds = len / sizeof(int);



            for (i = 0; i < numfds; i++)

                fd[i] = tswap32(target_fd[i]);

        }



        cmsg = CMSG_NXTHDR(msgh, cmsg);

        target_cmsg = TARGET_CMSG_NXTHDR(target_msgh, target_cmsg);

    }

    unlock_user(target_cmsg, target_cmsg_addr, 0);

 the_end:

    msgh->msg_controllen = space;

    return 0;

}
