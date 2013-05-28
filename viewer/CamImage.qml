/* Single camera view. Uses the current frame image as background, and adds
   target information. A status bar shows image name etc.
*/

import QtQuick 1.0

Item {
    property alias source: bg_img.source
    property alias targets: target_marks.model
    
    Text {
        id: status_line
        width: parent.width
        text: "Camera " + (index + 1) + " (" + file.split('/').reverse()[0] + ")"
    }
    Image { 
        id: bg_img
        width: parent.width
        anchors.top: status_line.bottom
        anchors.bottom: parent.bottom

        Repeater {
            id: target_marks
            delegate: Rectangle {
                x: posx/bg_img.sourceSize.width * bg_img.width - width/2.
                y: posy/bg_img.sourceSize.height * bg_img.height - height/2.
                
                width: 4; height: 4;
                radius: 4;
                color: "red"
            }
        }
    }
}
