static int configuration_post_load(void *opaque, int version_id)

{

    SaveState *state = opaque;

    const char *current_name = MACHINE_GET_CLASS(current_machine)->name;



    if (strncmp(state->name, current_name, state->len) != 0) {

        error_report("Machine type received is '%s' and local is '%s'",

                     state->name, current_name);

        return -EINVAL;

    }

    return 0;

}
