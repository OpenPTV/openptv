/* Implements a checkbox which changes its appearance (background and text
   weight) depending on state.
*/

import QtQuick 1.0

Rectangle {
    color: "red"
    id: opt_box
    property bool checked: false
    state: "unchecked"
    
    property alias text: caption.text
    height: caption.height
    width: caption.width
    
    Text {
        id: caption
        anchors.left: parent.left
        
        MouseArea {
            anchors.fill: parent
            onClicked: checked = !checked
        }
    }
    
    states: [
        State {
            name: "unchecked"
            PropertyChanges {target: opt_box; color: "grey"}
            PropertyChanges {target: caption; font.bold: false}
        },
        State {
            name: "checked"
            PropertyChanges {target: opt_box; color: "green"}
            PropertyChanges {target: caption; font.bold: true}
        }
    ]
    
    onCheckedChanged: state = (checked ? "checked" : "unchecked")
}

