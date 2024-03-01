from datasets.dfdc import DFDC
from datasets.ravdess import RAVDESS

def get_training_set(opt, spatial_transform=None, audio_transform=None):
    assert opt.dataset in ['RAVDESS', 'DFDC'], print('Unsupported dataset: {}'.format(opt.dataset))

    if opt.dataset == 'RAVDESS':
        training_data = RAVDESS(
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform, data_type='audiovisual', audio_transform=audio_transform)

    if opt.dataset == 'DFDC':
        training_data = DFDC(
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform, data_type='audiovisual', audio_transform=audio_transform)

    return training_data


def get_validation_set(opt, spatial_transform=None, audio_transform=None):
    assert opt.dataset in ['RAVDESS', 'DFDC'], print('Unsupported dataset: {}'.format(opt.dataset))

    if opt.dataset == 'RAVDESS':
        validation_data = RAVDESS(
            opt.annotation_path,
            'validation',
            spatial_transform=spatial_transform, data_type = 'audiovisual', audio_transform=audio_transform)

    if opt.dataset == 'DFDC':
        validation_data = DFDC(
            opt.annotation_path,
            'validation',
            spatial_transform=spatial_transform, data_type = 'audiovisual', audio_transform=audio_transform)

    return validation_data


def get_test_set(opt, spatial_transform=None, audio_transform=None):
    assert opt.dataset in ['RAVDESS', 'DFDC'], print('Unsupported dataset: {}'.format(opt.dataset))
    assert opt.test_subset in ['val', 'test']

    if opt.test_subset == 'val':
        subset = 'validation'
    elif opt.test_subset == 'test':
        subset = 'testing'
    if opt.dataset == 'RAVDESS':
        test_data = RAVDESS(
            opt.annotation_path,
            subset,
            spatial_transform=spatial_transform, data_type='audiovisual',audio_transform=audio_transform)

    if opt.dataset == 'DFDC':
        test_data = DFDC(
            opt.annotation_path,
            subset,
            spatial_transform=spatial_transform, data_type='audiovisual',audio_transform=audio_transform)

    return test_data
