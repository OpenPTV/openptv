
import QtQuick 1.0

Rectangle {
    property alias image: img.source
    signal clicked
    width: 16
    height: 16
    color: "lightgrey"
    border.width: 1
    
    MouseArea {
        anchors.fill: parent
        onClicked: parent.clicked()
    }
    
    Image {
        x: 1; y: 1
        id: img
    }
}

