bool migration_is_blocked(Error **errp)

{

    if (qemu_savevm_state_blocked(errp)) {

        return true;

    }



    if (migration_blockers) {

        *errp = error_copy(migration_blockers->data);

        return true;

    }



    return false;

}
