#
# This file is part of the AffBio package for clustering of
# biomolecular structures.
#
# Copyright (c) 2015-2016, by Arthur Zalevsky <aozalevsky@fbb.msu.ru>
#
# AffBio is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# AffBio is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public
# License along with AffBio; if not, see
# http://www.gnu.org/licenses, or write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
#

# -*- coding: UTF-8 -*-

import os
import re
import subprocess


class AffRender(object):

    def __init__(
            self,
            pdb_list=None,
            output=None,
            nums=list(),
            draw_nums=False,
            guess_nums=False,
            bcolor=False,
            lowt=10,
            width=640, height=480,
            moltype="general",
            clear=False,
            *args, **kwargs):

        self.models = pdb_list

        if output is None:
            self.out = 'out.png'
        else:
            self.out = output

        self.nums = nums
        self.draw_nums = draw_nums
        self.guess_nums = guess_nums

        if self.draw_nums and not self.guess_nums:
            if len(self.nums) != len(self.models):
                raise(Exception("Numbers of models and nums are different"))

        self.bcolor = False

        self.width = width
        self.height = height

        self.moltype = moltype

        self.clear = clear

        self.pymol = self.init_pymol()

        self.process_models()

        if bcolor is True:
            self.bcolor = bcolor
            filename, file_extension = os.path.splitext(self.out)
            self.out = filename + '_color.png'
            self.process_models()

    @staticmethod
    def init_pymol():
        import __main__
        __main__.pymol_argv = ['pymol', '-qc']
        import pymol
        pymol.finish_launching()
        return pymol

    def setup_scene(self):

        self.pymol.cmd.set("ambient", '0.00000')
        self.pymol.cmd.set("antialias", 4)
        self.pymol.cmd.set("light_count", 1)
        self.pymol.cmd.set("ray_shadow", 'off')
        self.pymol.cmd.set("reflect_power", '0.10000')
        self.pymol.cmd.set("spec_power", '0.00000')
        self.pymol.cmd.set("specular", '0.00000')
        self.pymol.cmd.set("orthoscopic", 1)

        # self.pymol.cmd.bg_color("white")
        self.pymol.cmd.set("opaque_background", 0)

    @staticmethod
    def tile(images, out, direction="h"):

        tile_format_d = {
            'h': "%dx",
            'v': "x%d"}

        tile_format = tile_format_d[direction] % len(images)

        call = [
            'montage',
            '-mode', 'Concatenate',
            '-background', 'none',
            '-tile', tile_format]
        call.extend(images)
        call.append(out)

        subprocess.call(call)

    # DNA origami specific part ###

    @staticmethod
    def is_backbone(i, j):
        return True if abs(i - j) == 1 else False

    @staticmethod
    def is_crossover(i, j):
        return True if abs(i - j) > 1 else False

    def draw_backbone(self):
        # Get total number of atoms
        # N = self.pymol.cmd.count_atoms()

        # Show backbone with sticks
        # for i in range(1, N):
        #   self.pymol.cmd.select('bck', 'resi %d+%d' % (i, i + 1))
        #   self.pymol.cmd.show('sticks', 'bck')

        # Set sticks width
        self.pymol.cmd.show("sticks")
        self.pymol.cmd.set("stick_radius", 1.5)

        # Find single stranded regions
        space = {"single": []}
        self.pymol.cmd.iterate("resn S*", "single.append(resi)", space=space)
        single = map(int, space["single"])

        # Set stich width for single stranded region
        # Only change stick radius for contigious regions in backbone
        # and do not touch crossovers
        if len(single) >= 2:
            for s in range(len(single) - 1):
                i = single[s]
                j = single[s + 1]
                if self.is_backbone(i, j):
                    self.pymol.cmd.set_bond(
                        "stick_radius", 0.5,
                        "i. %d" % i, "i. %d" % j)

        self.pymol.cmd.color('black')
        # Color bacbone according to B-factors
        if self.bcolor is True:
            self.pymol.cmd.spectrum("b")

    def draw_crossovers(self, fname):
        """ Read all CONECT from model file and draw all non-backbone
        bonds with dashes"""

        def get_ind(line):
            line = line.strip()
            line = re.sub('\s+', ';', line)
            i, j = map(int, line.split(";")[1:3])
            return i, j

        def is_bond(line):
            return True if re.match('CONECT', line) else False

        with open(fname, 'r') as f:
            bonds = f.readlines()

        crossover = []
        backbone = []

        for b in bonds:
            if is_bond(b):
                i, j = get_ind(b)
                if self.is_crossover(i, j):
                    crossover.append((i, j))
                elif self.is_backbone(i, j):
                    backbone.append((i, j))

        for b in crossover:
            i, j = b
            self.pymol.cmd.unbond("i. %d" % i, "i. %d" % j)
            self.pymol.cmd.distance("i. %d" % i, "i. %d" % j)
        self.pymol.cmd.hide("labels")

        self.pymol.cmd.set("dash_color", "gray80")
        self.pymol.cmd.set("dash_gap", 0)
        self.pymol.cmd.set("dash_length", 4)
        self.pymol.cmd.set("dash_radius", 1.0)

    def draw_nucleic_acid(self):
        self.pymol.cmd.hide("everything")
        self.pymol.cmd.show("lines")
        # self.pymol.cmd.show("cartoon")
        # self.pymol.cmd.set("cartoon_nucleic_acid_mode", 1)
        # self.pymol.cmd.set("cartoon_tube_radius", 0.1)
        # self.pymol.cmd.set("cartoon_ring_finder", 2)
        # self.pymol.cmd.set("cartoon_ring_mode", 2)
        # self.pymol.cmd.set("cartoon_flat_sheets", 0)

        # Color bacbone according to B-factors
        self.pymol.cmd.spectrum("b")

    @classmethod
    def gen_label(self, basename="gg", num=100, width=640, height=480):

        lwidth = int(0.2 * width)  # 20% - empirically
#        border = int(0.05 * lwidth)  # Border is 5% of label width

        name = self.gen_name(basename, 0)

        call = [
            "convert",
            "-transparent", "white",
            # "-background", "white",
            # "-bordercolor", "white",
            "-size",
            # "%dx%d" % (lwidth - 2 * border, height - 2 * border),
            "%dx%d" % (lwidth, height),
            # "-border", "%d" % border,
            "-gravity", "East",
            "-pointsize", "%d" % int(lwidth / 3),
            "caption:%d%%" % num,
            name]

        subprocess.call(call)

        return name

    def ray(self, name, width=640, height=480):
        # pymol.cmd.zoom("all", 100)
        self.pymol.cmd.zoom("all", 20)
        self.pymol.cmd.ray(width, height)
        self.pymol.cmd.save(name)

    @staticmethod
    def gen_name(basename, number):
        return "%s_%d.png" % (basename, number)

    def ray_poses(self, basename="gg", width=640, height=480):

        images = []

        name = self.gen_name(basename, 1)
        self.pymol.cmd.orient()
        self.ray(name, width, height)
        images.append(name)

        name = self.gen_name(basename, 2)
        self.pymol.cmd.rotate("x", 90)
        self.ray(name, width, height)
        images.append(name)

        name = self.gen_name(basename, 3)
        self.pymol.cmd.rotate("y", 90)
        self.ray(name, width, height)
        images.append(name)

        return images

    def process_model(self, index):

        model = self.models[index]

        # basename = model.replace('.pdb', '.png')
        basename = model[:-4] + '_tmp'

        if self.bcolor:
            basename += '_color'

        self.pymol.cmd.reinitialize()
        # Desired pymol commands here to produce and save figures
        self.setup_scene()

        self.pymol.cmd.load(model)

        if self.moltype == "general":
            self.draw_nucleic_acid()
        elif self.moltype == "origami":
            self.draw_crossovers(model)
            self.draw_backbone()

        images = list()

        if self.draw_nums:
            if self.guess_nums:
                num = self.get_num(model)
            else:
                num = self.nums[index]

            label = self.gen_label(
                basename, num, width=self.width, height=self.height)
            images.append(label)

        poses = self.ray_poses(basename, width=self.width, height=self.height)

        images.extend(poses)

        name = basename + '.png'
        self.tile(images=images, out=name)

        if self.clear:
            map(os.remove, images)

        return name

    @staticmethod
    def get_num(model):
        """ Get representativeness of current model """
        try:
            # try to parse filename like frameXXXX_aff_YY.pdb
            num = re.search('aff_(\d+)', model).groups()[0]
        except:
            # if no, set default num
            raise(Exception("Unable to get num from filename"))
        return int(num)

    def process_models(self):

        images = list()

        for i in range(len(self.models)):

            # now you can call it directly with basename
            image = self.process_model(i)
            images.append(image)

        self.tile(images, self.out, direction='v')

        if self.clear:
            map(os.remove, images)
