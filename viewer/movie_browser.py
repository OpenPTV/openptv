# A command-line tool for viewing four-camera movie directories as animation.
# Mostly loads the appropriate QML component and sets its properties from
# command-line arguments.

import sys, re
from PySide import QtCore
from PySide.QtGui import QApplication
from PySide.QtDeclarative import QDeclarativeView

from tracking_framebuf import read_targets

class TargetListModel(QtCore.QAbstractListModel):
    def __init__(self, parent, targets):
        QtCore.QAbstractListModel.__init__(self, parent)
        self._targets = targets
        self.setRoleNames({0: 'posx', 1: 'posy'})
 
    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self._targets)
 
    def data(self, index, role):
        if not index.isValid():
            return
        
        target = self._targets[index.row()]
        if role == 0:
            return target['x']
        elif role == 1:
            return target['y']

class ImageExplorer(QtCore.QObject):
    def __init__(self, name_template):
        QtCore.QObject.__init__(self)
        self._tmpl = name_template
    
    @QtCore.Slot(int, int, result=QtCore.QObject)
    def image_targets(self, frame, cam):
        tmpl = re.sub('%1', '%d', self._tmpl)
        tmpl = re.sub('%2', '', tmpl)
        targets = read_targets(tmpl % (cam + 1), frame)
        
        return TargetListModel(self, 
            [dict(zip(['x', 'y'], targets[t].pos())) \
                for t in xrange(len(targets))] )

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("template", help="Temlate for image file name, containing "
    "%%1 where camera number should be inserted, and %%2 where frame number "
    "should be inserted. Example: my_scene/cam%%1.%%2")
parser.add_argument("first", type=int, help="Number of first frame")
parser.add_argument("last", type=int, help="Number of last frame")
parser.add_argument("--rate", type=int, default=100,
    help="Frame rate in frames per second")
args = parser.parse_args()

app = QApplication(sys.argv)
view = QDeclarativeView()
view.setSource(QtCore.QUrl('MovieBrowser.qml'))

view.rootContext().setContextProperty("image_explorer", ImageExplorer(args.template))
view.rootObject().setProperty("scene_template", args.template)
view.rootObject().setProperty("first_frame", args.first)
view.rootObject().setProperty("last_frame", args.last)
view.rootObject().setProperty("frame_rate", args.rate)

# Respond to frame change signals by possibly loading targets:


view.show()
sys.exit(app.exec_())

