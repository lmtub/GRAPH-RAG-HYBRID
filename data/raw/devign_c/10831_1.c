static void drive_backup_abort(BlkActionState *common)

{

    DriveBackupState *state = DO_UPCAST(DriveBackupState, common, common);

    BlockDriverState *bs = state->bs;



    /* Only cancel if it's the job we started */

    if (bs && bs->job && bs->job == state->job) {

        block_job_cancel_sync(bs->job);

    }

}
