# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import oss2
import os
import six
import unittest

from tensorboard.compat.tensorflow_stub import errors
from tensorboard.compat.tensorflow_stub.io import gfile


_DEFAULT_TOP_DIRECTORY="top_dir"


class GFileTest(unittest.TestCase):
    def testExists(self):
        temp_dir = self._CreateDeepOSSStructure()
        ckpt_path = self._PathJoin(temp_dir, "model.ckpt")
        self.assertTrue(gfile.exists(temp_dir))
        self.assertTrue(gfile.exists(ckpt_path))

    def testGlob(self):
        temp_dir = self._CreateDeepOSSStructure()
        # S3 glob includes subdirectory content, which standard
        # filesystem does not. However, this is good for perf.
        expected = [
            "a.tfevents.1",
            "bar/b.tfevents.1",
            "bar/baz/c.tfevents.1",
            "bar/baz/d.tfevents.1",
            "bar/quux/some_flume_output.txt",
            "bar/quux/some_more_flume_output.txt",
            "bar/red_herring.txt",
            "model.ckpt",
            "quuz/e.tfevents.1",
            "quuz/garply/corge/g.tfevents.1",
            "quuz/garply/f.tfevents.1",
            "quuz/garply/grault/h.tfevents.1",
            "waldo/fred/i.tfevents.1",
        ]
        expected_listing = [self._PathJoin(temp_dir, f) for f in expected]
        gotten_listing = gfile.glob(self._PathJoin(temp_dir, "*"))
        six.assertCountEqual(
            self,
            expected_listing,
            gotten_listing,
            "Files must match. Expected %r. Got %r."
            % (expected_listing, gotten_listing),
        )

    def testIsdir(self):
        temp_dir = self._CreateDeepOSSStructure()
        self.assertTrue(gfile.isdir(temp_dir))

    def testListdir(self):
        temp_dir = self._CreateDeepOSSStructure()
        # self._CreateDeepOSSStructure(temp_dir)
        expected_files = [
            # Empty directory also returned
            'foo',
            "bar",
            "quuz",
            "a.tfevents.1",
            "model.ckpt",
            "waldo",
        ]
        gotten_files = gfile.listdir(temp_dir)
        six.assertCountEqual(self, expected_files, gotten_files)
        
    def testMakeDirs(self):
        temp_dir = self._CreateDeepOSSStructure()
        new_dir = self._PathJoin(temp_dir, "newdir", "subdir", "subsubdir")
        gfile.makedirs(new_dir)
        self.assertTrue(gfile.isdir(new_dir))

    def testMakeDirsAlreadyExists(self):
        temp_dir = self._CreateDeepOSSStructure()
        new_dir = self._PathJoin(temp_dir, "bar", "baz")
        with self.assertRaises(errors.AlreadyExistsError):
            gfile.makedirs(new_dir)

    def testWalk(self):
        temp_dir = self._CreateDeepOSSStructure()
        self._CreateDeepOSSStructure(temp_dir)
        expected = [
            ["", ["a.tfevents.1", "model.ckpt",]],
            # Empty directory also returned
            ['foo', []],
            ['newdir', []],
            ['newdir/subdir', []],
            ['newdir/subdir/subsubdir', []],
            ["bar", ["b.tfevents.1", "red_herring.txt",]],
            ["bar/baz", ["c.tfevents.1", "d.tfevents.1",]],
            [
                "bar/quux",
                ["some_flume_output.txt", "some_more_flume_output.txt",],
            ],
            ["quuz", ["e.tfevents.1",]],
            ["quuz/garply", ["f.tfevents.1",]],
            ["quuz/garply/corge", ["g.tfevents.1",]],
            ["quuz/garply/grault", ["h.tfevents.1",]],
            ["waldo", []],
            ["waldo/fred", ["i.tfevents.1",]],
        ]
        for pair in expected:
            # If this is not the top-level directory, prepend the high-level
            # directory.
            pair[0] = self._PathJoin(temp_dir, pair[0]) if pair[0] else temp_dir
        gotten = gfile.walk(temp_dir)
        self._CompareFilesPerSubdirectory(expected, gotten)

    def testStat(self):
        ckpt_content = "asdfasdfasdffoobarbuzz"
        temp_dir = self._CreateDeepOSSStructure(ckpt_content=ckpt_content)
        ckpt_path = self._PathJoin(temp_dir, "model.ckpt")
        ckpt_stat = gfile.stat(ckpt_path)
        self.assertEqual(ckpt_stat.length, len(ckpt_content))
        bad_ckpt_path = self._PathJoin(temp_dir, "bad_model.ckpt")
        with self.assertRaises(errors.NotFoundError):
            gfile.stat(bad_ckpt_path)

    def testRead(self):
        ckpt_content = "asdfasdfasdffoobarbuzz"
        temp_dir = self._CreateDeepOSSStructure(ckpt_content=ckpt_content)
        ckpt_path = self._PathJoin(temp_dir, "model.ckpt")
        with gfile.GFile(ckpt_path, "r") as f:
            f.buff_chunk_size = 4  # Test buffering by reducing chunk size
            ckpt_read = f.read()
            self.assertEqual(ckpt_content, ckpt_read)

    def testReadLines(self):
        ckpt_lines = (
            [u"\n"] + [u"line {}\n".format(i) for i in range(10)] + [u" "]
        )
        ckpt_content = u"".join(ckpt_lines)
        temp_dir = self._CreateDeepOSSStructure(ckpt_content=ckpt_content)
        ckpt_path = self._PathJoin(temp_dir, "model.ckpt")
        with gfile.GFile(ckpt_path, "r") as f:
            f.buff_chunk_size = 4  # Test buffering by reducing chunk size
            ckpt_read_lines = list(f)
            self.assertEqual(ckpt_lines, ckpt_read_lines)

    def testReadWithOffset(self):
        ckpt_content = "asdfasdfasdffoobarbuzz"
        ckpt_b_content = b"asdfasdfasdffoobarbuzz"
        temp_dir = self._CreateDeepOSSStructure(ckpt_content=ckpt_content)
        ckpt_path = self._PathJoin(temp_dir, "model.ckpt")
        with gfile.GFile(ckpt_path, "r") as f:
            f.buff_chunk_size = 4  # Test buffering by reducing chunk size
            ckpt_read = f.read(12)
            self.assertEqual("asdfasdfasdf", ckpt_read)
            ckpt_read = f.read(6)
            self.assertEqual("foobar", ckpt_read)
            ckpt_read = f.read(1)
            self.assertEqual("b", ckpt_read)
            ckpt_read = f.read()
            self.assertEqual("uzz", ckpt_read)
            ckpt_read = f.read(1000)
            self.assertEqual("", ckpt_read)
        with gfile.GFile(ckpt_path, "rb") as f:
            ckpt_read = f.read()
            self.assertEqual(ckpt_b_content, ckpt_read)

    def testWrite(self):
        temp_dir = self._CreateDeepOSSStructure()
        ckpt_path = os.path.join(temp_dir, "model2.ckpt")
        ckpt_content = u"asdfasdfasdffoobarbuzz"
        with gfile.GFile(ckpt_path, "w") as f:
            f.write(ckpt_content)
        with gfile.GFile(ckpt_path, "r") as f:
            ckpt_read = f.read()
            self.assertEqual(ckpt_content, ckpt_read)

    def testOverwrite(self):
        temp_dir = self._CreateDeepOSSStructure()
        ckpt_path = os.path.join(temp_dir, "model2.ckpt")
        ckpt_content = u"asdfasdfasdffoobarbuzz"
        return True
        with gfile.GFile(ckpt_path, "w") as f:
            f.write(u"original")
        with gfile.GFile(ckpt_path, "w") as f:
            f.write(ckpt_content)
        with gfile.GFile(ckpt_path, "r") as f:
            ckpt_read = f.read()
            self.assertEqual(ckpt_content, ckpt_read)

    def testWriteMultiple(self):
        temp_dir = self._CreateDeepOSSStructure()
        ckpt_path = os.path.join(temp_dir, "model2.ckpt")
        ckpt_content = u"asdfasdfasdffoobarbuzz" * 5
        with gfile.GFile(ckpt_path, "w") as f:
            for i in range(0, len(ckpt_content), 3):
                f.write(ckpt_content[i : i + 3])
                # Test periodic flushing of the file
                if i % 9 == 0:
                    f.flush()
        with gfile.GFile(ckpt_path, "r") as f:
            ckpt_read = f.read()
            self.assertEqual(ckpt_content, ckpt_read)

    def testWriteEmpty(self):
        temp_dir = self._CreateDeepOSSStructure()
        ckpt_path = os.path.join(temp_dir, "model2.ckpt")
        ckpt_content = u""
        with gfile.GFile(ckpt_path, "w") as f:
            f.write(ckpt_content)
        with gfile.GFile(ckpt_path, "r") as f:
            ckpt_read = f.read()
            self.assertEqual(ckpt_content, ckpt_read)

    def testWriteBinary(self):
        temp_dir = self._CreateDeepOSSStructure()
        ckpt_path = os.path.join(temp_dir, "model.ckpt")
        ckpt_content = b"asdfasdfasdffoobarbuzz"
        with gfile.GFile(ckpt_path, "wb") as f:
            f.write(ckpt_content)
        with gfile.GFile(ckpt_path, "rb") as f:
            ckpt_read = f.read()
            self.assertEqual(ckpt_content, ckpt_read)

    def testWriteMultipleBinary(self):
        temp_dir = self._CreateDeepOSSStructure()
        ckpt_path = os.path.join(temp_dir, "model2.ckpt")
        ckpt_content = b"asdfasdfasdffoobarbuzz" * 5
        with gfile.GFile(ckpt_path, "wb") as f:
            for i in range(0, len(ckpt_content), 3):
                f.write(ckpt_content[i : i + 3])
                # Test periodic flushing of the file
                if i % 9 == 0:
                    f.flush()
        with gfile.GFile(ckpt_path, "rb") as f:
            ckpt_read = f.read()
            self.assertEqual(ckpt_content, ckpt_read)

    @classmethod
    def tearDownClass(cls):
        auth = oss2.Auth(os.environ["OSS_ACCESS_KEY_ID"],
                         os.environ["OSS_ACCESS_KEY_SECRET"])
        endpoint = os.environ["OSS_ENDPOINT"]
        bucket_name = os.environ["OSS_TEST_BUCKET"]
        bucket = oss2.Bucket(auth, endpoint, bucket_name)

        top_directory = os.environ.get("OSS_TEST_DIR", _DEFAULT_TOP_DIRECTORY)
        for obj in oss2.ObjectIterator(bucket, prefix=top_directory + "/"):
            bucket.delete_object(obj.key)

    def _PathJoin(self, *args):
        """Join directory and path with slash and not local separator."""
        return "/".join(args)

    def _CreateDeepOSSStructure(
        self,
        ckpt_content="",
    ):
        """Creates a reasonable deep structure of OSS subdirectories with files.

        Args:
          top_directory: The path of the top level OSS directory in which
            to create the directory structure. Defaults to 'top_dir'.
          ckpt_content: The content to put into model.ckpt. Default to ''.
          region_name: The OSS region name. Defaults to 'us-east-1'.
          bucket_name: The OSS bucket name. Defaults to 'test'.

        Returns:
          OSS URL of the top directory in the form 's3://bucket/path'
        """
        top_directory = os.environ.get("OSS_TEST_DIR", _DEFAULT_TOP_DIRECTORY)
        bucket_name = os.environ["OSS_TEST_BUCKET"]
        oss_top_url = "oss://{}/{}".format(bucket_name, top_directory)
        auth = oss2.Auth(os.environ['OSS_ACCESS_KEY_ID'],
                         os.environ['OSS_ACCESS_KEY_SECRET'])
        endpoint = os.environ['OSS_ENDPOINT']
        bucket = oss2.Bucket(auth, endpoint, bucket_name)

        # Add a few subdirectories.
        directory_names = (
            # An empty directory.
            "foo",
            # A directory with an events file (and a text file).
            "bar",
            # A deeper directory with events files.
            "bar/baz",
            # A non-empty subdir that lacks event files (should be ignored).
            "bar/quux",
            # This 3-level deep set of subdirectories tests logic that replaces
            # the full glob string with an absolute path prefix if there is
            # only 1 subdirectory in the final mapping.
            "quuz/garply",
            "quuz/garply/corge",
            "quuz/garply/grault",
            # A directory that lacks events files, but contains a subdirectory
            # with events files (first level should be ignored, second level
            # should be included).
            "waldo",
            "waldo/fred",
        )
        bucket.put_object(key=top_directory + "/", data="")
        for directory_name in directory_names:
            # Add an end slash
            path = top_directory + "/" + directory_name + "/"
            # Create an empty object so the location exists
            bucket.put_object(key=path, data="")

        # Add a few files to the directory.
        file_names = (
            "a.tfevents.1",
            "model.ckpt",
            "bar/b.tfevents.1",
            "bar/red_herring.txt",
            "bar/baz/c.tfevents.1",
            "bar/baz/d.tfevents.1",
            "bar/quux/some_flume_output.txt",
            "bar/quux/some_more_flume_output.txt",
            "quuz/e.tfevents.1",
            "quuz/garply/f.tfevents.1",
            "quuz/garply/corge/g.tfevents.1",
            "quuz/garply/grault/h.tfevents.1",
            "waldo/fred/i.tfevents.1",
        )
        for file_name in file_names:
            # Add an end slash
            path = top_directory + "/" + file_name
            if file_name == "model.ckpt":
                content = ckpt_content
            else:
                content = ""
            bucket.put_object(key=path, data=content)
        return oss_top_url

    def _CompareFilesPerSubdirectory(self, expected, gotten):
        """Compares iterables of (subdirectory path, list of absolute paths)

        Args:
          expected: The expected iterable of 2-tuples.
          gotten: The gotten iterable of 2-tuples.
        """
        expected_directory_to_files = {
            result[0]: list(result[1]) for result in expected
        }
        gotten_directory_to_files = {
            # Note we ignore subdirectories and just compare files
            result[0]: list(result[2]) for result in gotten
        }
        six.assertCountEqual(
            self,
            expected_directory_to_files.keys(),
            gotten_directory_to_files.keys(),
        )

        for subdir, expected_listing in expected_directory_to_files.items():
            gotten_listing = gotten_directory_to_files[subdir]
            six.assertCountEqual(
                self,
                expected_listing,
                gotten_listing,
                "Files for subdir %r must match. Expected %r. Got %r."
                % (subdir, expected_listing, gotten_listing),
            )


if __name__ == "__main__":
    unittest.main()
