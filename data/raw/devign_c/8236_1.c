void qemu_savevm_state_begin(QEMUFile *f,

                             const MigrationParams *params)

{

    SaveStateEntry *se;

    int ret;



    trace_savevm_state_begin();

    QTAILQ_FOREACH(se, &savevm_state.handlers, entry) {

        if (!se->ops || !se->ops->set_params) {

            continue;

        }

        se->ops->set_params(params, se->opaque);

    }



    QTAILQ_FOREACH(se, &savevm_state.handlers, entry) {

        if (!se->ops || !se->ops->save_live_setup) {

            continue;

        }

        if (se->ops && se->ops->is_active) {

            if (!se->ops->is_active(se->opaque)) {

                continue;

            }

        }

        save_section_header(f, se, QEMU_VM_SECTION_START);



        ret = se->ops->save_live_setup(f, se->opaque);


        if (ret < 0) {

            qemu_file_set_error(f, ret);

            break;

        }

    }

}