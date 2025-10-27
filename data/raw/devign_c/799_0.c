int vmstate_register(int instance_id, const VMStateDescription *vmsd,

                     void *opaque)

{

    SaveStateEntry *se;



    se = qemu_malloc(sizeof(SaveStateEntry));

    pstrcpy(se->idstr, sizeof(se->idstr), vmsd->name);

    se->version_id = vmsd->version_id;

    se->section_id = global_section_id++;

    se->save_live_state = NULL;

    se->save_state = NULL;

    se->load_state = NULL;

    se->opaque = opaque;

    se->vmsd = vmsd;



    if (instance_id == -1) {

        se->instance_id = calculate_new_instance_id(vmsd->name);

    } else {

        se->instance_id = instance_id;

    }

    /* add at the end of list */

    TAILQ_INSERT_TAIL(&savevm_handlers, se, entry);

    return 0;

}
