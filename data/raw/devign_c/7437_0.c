static TargetFdDataFunc fd_trans_host_to_target_data(int fd)

{

    if (fd < target_fd_max && target_fd_trans[fd]) {

        return target_fd_trans[fd]->host_to_target_data;

    }

    return NULL;

}
