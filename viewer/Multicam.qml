
import QtQuick 1.0

Item {
    id: cams_view
    width: 600; height: 600
    
    property alias images: init_images
    ListModel {
        id: init_images
        ListElement { file: ""; target_list: "" }
        ListElement { file: ""; target_list: "" }
        ListElement { file: ""; target_list: "" }
        ListElement { file: ""; target_list: "" }
    }
    
    Grid {
        id: cam_grid
        rows: 2; columns: 2
        flow: Grid.LeftToRight
        anchors.fill: parent
        spacing: 2
        
        Repeater {
            model: init_images
            delegate: CamImage {
                source: file
                targets: target_list
                width: cam_grid.width / 2
                height: cam_grid.height / 2
            }
        }
    }
}

