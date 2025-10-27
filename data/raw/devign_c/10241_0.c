_syscall3(int,sys_faccessat,int,dirfd,const char *,pathname,int,mode)

#endif

#if defined(TARGET_NR_fchmodat) && defined(__NR_fchmodat)

_syscall3(int,sys_fchmodat,int,dirfd,const char *,pathname, mode_t,mode)

#endif

#if defined(TARGET_NR_fchownat) && defined(__NR_fchownat) && defined(USE_UID16)

_syscall5(int,sys_fchownat,int,dirfd,const char *,pathname,

          uid_t,owner,gid_t,group,int,flags)

#endif

#if (defined(TARGET_NR_fstatat64) || defined(TARGET_NR_newfstatat)) && \

        defined(__NR_fstatat64)

_syscall4(int,sys_fstatat64,int,dirfd,const char *,pathname,

          struct stat *,buf,int,flags)

#endif

#if defined(TARGET_NR_futimesat) && defined(__NR_futimesat)

_syscall3(int,sys_futimesat,int,dirfd,const char *,pathname,

         const struct timeval *,times)

#endif

#if (defined(TARGET_NR_newfstatat) || defined(TARGET_NR_fstatat64) ) && \

        defined(__NR_newfstatat)

_syscall4(int,sys_newfstatat,int,dirfd,const char *,pathname,

          struct stat *,buf,int,flags)

#endif

#if defined(TARGET_NR_linkat) && defined(__NR_linkat)

_syscall5(int,sys_linkat,int,olddirfd,const char *,oldpath,

      int,newdirfd,const char *,newpath,int,flags)

#endif

#if defined(TARGET_NR_mkdirat) && defined(__NR_mkdirat)

_syscall3(int,sys_mkdirat,int,dirfd,const char *,pathname,mode_t,mode)

#endif

#if defined(TARGET_NR_mknodat) && defined(__NR_mknodat)

_syscall4(int,sys_mknodat,int,dirfd,const char *,pathname,

          mode_t,mode,dev_t,dev)

#endif

#if defined(TARGET_NR_openat) && defined(__NR_openat)

_syscall4(int,sys_openat,int,dirfd,const char *,pathname,int,flags,mode_t,mode)

#endif

#if defined(TARGET_NR_readlinkat) && defined(__NR_readlinkat)

_syscall4(int,sys_readlinkat,int,dirfd,const char *,pathname,

          char *,buf,size_t,bufsize)

#endif

#if defined(TARGET_NR_renameat) && defined(__NR_renameat)

_syscall4(int,sys_renameat,int,olddirfd,const char *,oldpath,

          int,newdirfd,const char *,newpath)

#endif

#if defined(TARGET_NR_symlinkat) && defined(__NR_symlinkat)

_syscall3(int,sys_symlinkat,const char *,oldpath,

          int,newdirfd,const char *,newpath)

#endif

#if defined(TARGET_NR_unlinkat) && defined(__NR_unlinkat)

_syscall3(int,sys_unlinkat,int,dirfd,const char *,pathname,int,flags)

#endif

#if defined(TARGET_NR_utimensat) && defined(__NR_utimensat)

_syscall4(int,sys_utimensat,int,dirfd,const char *,pathname,

          const struct timespec *,tsp,int,flags)

#endif



#endif /* CONFIG_ATFILE */



#ifdef CONFIG_INOTIFY

#include <sys/inotify.h>



#if defined(TARGET_NR_inotify_init) && defined(__NR_inotify_init)

static int sys_inotify_init(void)

{

  return (inotify_init());

}
