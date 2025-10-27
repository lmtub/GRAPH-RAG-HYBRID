_syscall4(int,sys_utimensat,int,dirfd,const char *,pathname,

          const struct timespec *,tsp,int,flags)

#endif

#endif /* CONFIG_UTIMENSAT  */



#ifdef CONFIG_INOTIFY

#include <sys/inotify.h>



#if defined(TARGET_NR_inotify_init) && defined(__NR_inotify_init)

static int sys_inotify_init(void)

{

  return (inotify_init());

}
