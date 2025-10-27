static int do_fork(CPUArchState *env, unsigned int flags, abi_ulong newsp,

                   abi_ulong parent_tidptr, target_ulong newtls,

                   abi_ulong child_tidptr)

{

    CPUState *cpu = ENV_GET_CPU(env);

    int ret;

    TaskState *ts;

    CPUState *new_cpu;

    CPUArchState *new_env;

    sigset_t sigmask;



    /* Emulate vfork() with fork() */

    if (flags & CLONE_VFORK)

        flags &= ~(CLONE_VFORK | CLONE_VM);



    if (flags & CLONE_VM) {

        TaskState *parent_ts = (TaskState *)cpu->opaque;

        new_thread_info info;

        pthread_attr_t attr;



        ts = g_new0(TaskState, 1);

        init_task_state(ts);

        /* we create a new CPU instance. */

        new_env = cpu_copy(env);

        /* Init regs that differ from the parent.  */

        cpu_clone_regs(new_env, newsp);

        new_cpu = ENV_GET_CPU(new_env);

        new_cpu->opaque = ts;

        ts->bprm = parent_ts->bprm;

        ts->info = parent_ts->info;

        ts->signal_mask = parent_ts->signal_mask;



        if (flags & CLONE_CHILD_CLEARTID) {

            ts->child_tidptr = child_tidptr;

        }



        if (flags & CLONE_SETTLS) {

            cpu_set_tls (new_env, newtls);

        }



        /* Grab a mutex so that thread setup appears atomic.  */

        pthread_mutex_lock(&clone_lock);



        memset(&info, 0, sizeof(info));

        pthread_mutex_init(&info.mutex, NULL);

        pthread_mutex_lock(&info.mutex);

        pthread_cond_init(&info.cond, NULL);

        info.env = new_env;

        if (flags & CLONE_CHILD_SETTID) {

            info.child_tidptr = child_tidptr;

        }

        if (flags & CLONE_PARENT_SETTID) {

            info.parent_tidptr = parent_tidptr;

        }



        ret = pthread_attr_init(&attr);

        ret = pthread_attr_setstacksize(&attr, NEW_STACK_SIZE);

        ret = pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);

        /* It is not safe to deliver signals until the child has finished

           initializing, so temporarily block all signals.  */

        sigfillset(&sigmask);

        sigprocmask(SIG_BLOCK, &sigmask, &info.sigmask);



        ret = pthread_create(&info.thread, &attr, clone_func, &info);

        /* TODO: Free new CPU state if thread creation failed.  */



        sigprocmask(SIG_SETMASK, &info.sigmask, NULL);

        pthread_attr_destroy(&attr);

        if (ret == 0) {

            /* Wait for the child to initialize.  */

            pthread_cond_wait(&info.cond, &info.mutex);

            ret = info.tid;

        } else {

            ret = -1;

        }

        pthread_mutex_unlock(&info.mutex);

        pthread_cond_destroy(&info.cond);

        pthread_mutex_destroy(&info.mutex);

        pthread_mutex_unlock(&clone_lock);

    } else {

        /* if no CLONE_VM, we consider it is a fork */

        if ((flags & ~(CSIGNAL | CLONE_NPTL_FLAGS2)) != 0) {

            return -TARGET_EINVAL;

        }



        if (block_signals()) {

            return -TARGET_ERESTARTSYS;

        }



        fork_start();

        ret = fork();

        if (ret == 0) {

            /* Child Process.  */

            rcu_after_fork();

            cpu_clone_regs(env, newsp);

            fork_end(1);

            /* There is a race condition here.  The parent process could

               theoretically read the TID in the child process before the child

               tid is set.  This would require using either ptrace

               (not implemented) or having *_tidptr to point at a shared memory

               mapping.  We can't repeat the spinlock hack used above because

               the child process gets its own copy of the lock.  */

            if (flags & CLONE_CHILD_SETTID)

                put_user_u32(gettid(), child_tidptr);

            if (flags & CLONE_PARENT_SETTID)

                put_user_u32(gettid(), parent_tidptr);

            ts = (TaskState *)cpu->opaque;

            if (flags & CLONE_SETTLS)

                cpu_set_tls (env, newtls);

            if (flags & CLONE_CHILD_CLEARTID)

                ts->child_tidptr = child_tidptr;

        } else {

            fork_end(0);

        }

    }

    return ret;

}
