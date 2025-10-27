static void fill_thread_info(struct elf_note_info *info, const CPUState *env)

{

    TaskState *ts = (TaskState *)env->opaque;

    struct elf_thread_status *ets;



    ets = qemu_mallocz(sizeof (*ets));

    ets->num_notes = 1; /* only prstatus is dumped */

    fill_prstatus(&ets->prstatus, ts, 0);

    elf_core_copy_regs(&ets->prstatus.pr_reg, env);

    fill_note(&ets->notes[0], "CORE", NT_PRSTATUS, sizeof (ets->prstatus),

        &ets->prstatus);



    TAILQ_INSERT_TAIL(&info->thread_list, ets, ets_link);



    info->notes_size += note_size(&ets->notes[0]);

}
