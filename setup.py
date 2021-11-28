# Copyright 2021 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A setuptools based setup module for magenta."""
import subprocess
from distutils.command.build import build as _build

import setuptools
from setuptools import find_packages
from setuptools import setup


class build(_build):  # pylint: disable=invalid-name
  """A build command class that will be invoked during package install.
  The package built using the current setup.py will be staged and later
  installed in the worker using `pip install package'. This class will be
  instantiated during install for this specific scenario and will trigger
  running the custom commands specified.
  """
  sub_commands = _build.sub_commands + [('CustomCommands', None)]

CUSTOM_COMMANDS = [['sudo', 'apt-get', 'install', '-y', 'build-essential', 'libasound2-dev', 'libjack-dev', 'portaudio19-dev']]


class CustomCommands(setuptools.Command):
  """A setuptools Command class able to run arbitrary commands."""
  def initialize_options(self):
    pass

  def finalize_options(self):
    pass

  def RunCustomCommand(self, command_list):
    print('Running command: %s' % command_list)
    p = subprocess.Popen(
        command_list,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    # Can use communicate(input='y\n'.encode()) if the command run requires
    # some confirmation.
    stdout_data, _ = p.communicate()
    print('Command output: %s' % stdout_data)
    if p.returncode != 0:
      raise RuntimeError(
          'Command %s failed: exit code: %s' % (command_list, p.returncode))

  def run(self):
    for command in CUSTOM_COMMANDS:
      self.RunCustomCommand(command)
# Bit of a hack to parse the version string stored in version.py without
# executing __init__.py, which will end up requiring a bunch of dependencies to
# execute (e.g., tensorflow, pretty_midi, etc.).
# Makes the __version__ variable available.
with open('magenta/version.py') as in_file:
  exec(in_file.read())  # pylint: disable=exec-used

REQUIRED_PACKAGES = [
    'wheel',
    'kfac==0.2.0',
    'absl-py==0.12.0',
    'pyparsing==2.4.7',
    'dm-sonnet',
    # 'numpy==1.20.3',
    'numpy<1.20.0',
    'dill==0.3.1.1',
    # 'typing-extensions==3.10.0.2',
    'typing-extensions<3.8.0',
    'bokeh<2.0.0',
    'google-api-core<2.0.0dev',
    # tensor2tensor has a dependency on dopamine-rl, which we don't use.
    # pin to a version that doesn't require pygame installation because that
    # has too many external non-python dependencies.
    'dopamine-rl <= 3.0.1',
    'imageio',
    'librosa >= 0.6.2, < 0.7.0',
    'matplotlib >= 1.5.3',
    # 'matplotlib < 3',
    'mido == 1.2.6',
    'mir_eval >= 0.4',
    'note-seq',
    'numba < 0.50',  # temporary fix for librosa import
    'Pillow >= 3.4.2',
    'pretty_midi >= 0.2.6',
    'pygtrie >= 2.3',
    'python-rtmidi >= 1.1, < 1.2',  # 1.2 breaks us
    'scikit-image',
    'scipy >= 0.18.1',
    'six >= 1.12.0',
    'sk-video',
    'sox >= 1.3.7',
    'tensor2tensor',
    'tensorflow',
    'tensorflow-datasets',
    'tensorflow-probability==0.7.0',
    'tf_slim',
]

EXTRAS_REQUIRE = {
    'beam': [
        # 'apache-beam[gcp] >= 2.14.0',
        'apache-beam[gcp]==2.28.0'
    ],
    'onsets_frames_realtime': [
        'pyaudio',
        'colorama',
        'tflite',
    ],
    'test': [
        'pylint',
        'pytest',
    ]
}

# pylint:disable=line-too-long
CONSOLE_SCRIPTS = [
    'magenta.interfaces.midi.magenta_midi',
    'magenta.interfaces.midi.midi_clock',
    'magenta.models.arbitrary_image_stylization.arbitrary_image_stylization_evaluate',
    'magenta.models.arbitrary_image_stylization.arbitrary_image_stylization_train',
    'magenta.models.arbitrary_image_stylization.arbitrary_image_stylization_with_weights',
    'magenta.models.arbitrary_image_stylization.arbitrary_image_stylization_distill_mobilenet',
    'magenta.models.drums_rnn.drums_rnn_create_dataset',
    'magenta.models.drums_rnn.drums_rnn_generate',
    'magenta.models.drums_rnn.drums_rnn_train',
    'magenta.models.image_stylization.image_stylization_create_dataset',
    'magenta.models.image_stylization.image_stylization_evaluate',
    'magenta.models.image_stylization.image_stylization_finetune',
    'magenta.models.image_stylization.image_stylization_train',
    'magenta.models.image_stylization.image_stylization_transform',
    'magenta.models.improv_rnn.improv_rnn_create_dataset',
    'magenta.models.improv_rnn.improv_rnn_generate',
    'magenta.models.improv_rnn.improv_rnn_train',
    'magenta.models.gansynth.gansynth_train',
    'magenta.models.gansynth.gansynth_generate',
    'magenta.models.melody_rnn.melody_rnn_create_dataset',
    'magenta.models.melody_rnn.melody_rnn_generate',
    'magenta.models.melody_rnn.melody_rnn_train',
    'magenta.models.music_vae.music_vae_generate',
    'magenta.models.music_vae.music_vae_train',
    'magenta.models.nsynth.wavenet.nsynth_generate',
    'magenta.models.nsynth.wavenet.nsynth_save_embeddings',
    'magenta.models.onsets_frames_transcription.onsets_frames_transcription_create_dataset',
    'magenta.models.onsets_frames_transcription.onsets_frames_transcription_create_dataset_maps',
    'magenta.models.onsets_frames_transcription.onsets_frames_transcription_create_tfrecords',
    'magenta.models.onsets_frames_transcription.onsets_frames_transcription_infer',
    'magenta.models.onsets_frames_transcription.onsets_frames_transcription_train',
    'magenta.models.onsets_frames_transcription.onsets_frames_transcription_transcribe',
    'magenta.models.onsets_frames_transcription.realtime.onsets_frames_transcription_realtime',
    'magenta.models.performance_rnn.performance_rnn_create_dataset',
    'magenta.models.performance_rnn.performance_rnn_generate',
    'magenta.models.performance_rnn.performance_rnn_train',
    'magenta.models.pianoroll_rnn_nade.pianoroll_rnn_nade_create_dataset',
    'magenta.models.pianoroll_rnn_nade.pianoroll_rnn_nade_generate',
    'magenta.models.pianoroll_rnn_nade.pianoroll_rnn_nade_train',
    'magenta.models.polyphony_rnn.polyphony_rnn_create_dataset',
    'magenta.models.polyphony_rnn.polyphony_rnn_generate',
    'magenta.models.polyphony_rnn.polyphony_rnn_train',
    'magenta.models.rl_tuner.rl_tuner_train',
    'magenta.models.sketch_rnn.sketch_rnn_train',
    'magenta.scripts.convert_dir_to_note_sequences',
    'magenta.tensor2tensor.t2t_datagen',
    'magenta.tensor2tensor.t2t_decoder',
    'magenta.tensor2tensor.t2t_trainer',
]
# pylint:enable=line-too-long

setup(
    name='magenta',
    version=__version__,  # pylint: disable=undefined-variable
    description='Use machine learning to create art and music',
    long_description='',
    url='https://magenta.tensorflow.org/',
    author='Google Inc.',
    author_email='magenta-discuss@gmail.com',
    license='Apache 2',
    # PyPI package information.
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],
    keywords='tensorflow machine learning magenta music art',

    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRAS_REQUIRE,
    setup_requires=['wheel'],
    entry_points={
        'console_scripts': ['%s = %s:console_entry_point' % (n, p) for n, p in
                            ((s.split('.')[-1], s) for s in CONSOLE_SCRIPTS)],
    },

    include_package_data=True,
    package_data={
        'magenta': ['models/image_stylization/evaluation_images/*.jpg'],
    },
    # cmdclass={
    #     # Command class instantiated and run during pip install scenarios.
    #     'build': build,
    #     'CustomCommands': CustomCommands,
    # }
)
