int monitor_fdset_dup_fd_remove(int dup_fd)

{

    return monitor_fdset_dup_fd_find_remove(dup_fd, true);

}
