int64_t qmp_guest_get_time(Error **errp)

{

   int ret;

   qemu_timeval tq;

   int64_t time_ns;



   ret = qemu_gettimeofday(&tq);

   if (ret < 0) {

       error_setg_errno(errp, errno, "Failed to get time");

       return -1;

   }



   time_ns = tq.tv_sec * 1000000000LL + tq.tv_usec * 1000;

   return time_ns;

}
