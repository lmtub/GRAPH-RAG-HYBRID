static int fill_note_info(struct elf_note_info *info,

                          long signr, const CPUArchState *env)

{

#define NUMNOTES 3

    CPUState *cpu = ENV_GET_CPU((CPUArchState *)env);

    TaskState *ts = (TaskState *)cpu->opaque;

    int i;



    info->notes = g_malloc0(NUMNOTES * sizeof (struct memelfnote));

    if (info->notes == NULL)

        return (-ENOMEM);

    info->prstatus = g_malloc0(sizeof (*info->prstatus));

    if (info->prstatus == NULL)

        return (-ENOMEM);

    info->psinfo = g_malloc0(sizeof (*info->psinfo));

    if (info->prstatus == NULL)

        return (-ENOMEM);



    /*

     * First fill in status (and registers) of current thread

     * including process info & aux vector.

     */

    fill_prstatus(info->prstatus, ts, signr);

    elf_core_copy_regs(&info->prstatus->pr_reg, env);

    fill_note(&info->notes[0], "CORE", NT_PRSTATUS,

              sizeof (*info->prstatus), info->prstatus);

    fill_psinfo(info->psinfo, ts);

    fill_note(&info->notes[1], "CORE", NT_PRPSINFO,

              sizeof (*info->psinfo), info->psinfo);

    fill_auxv_note(&info->notes[2], ts);

    info->numnote = 3;



    info->notes_size = 0;

    for (i = 0; i < info->numnote; i++)

        info->notes_size += note_size(&info->notes[i]);



    /* read and fill status of all threads */

    cpu_list_lock();

    CPU_FOREACH(cpu) {

        if (cpu == thread_cpu) {

            continue;

        }

        fill_thread_info(info, (CPUArchState *)cpu->env_ptr);

    }

    cpu_list_unlock();



    return (0);

}
