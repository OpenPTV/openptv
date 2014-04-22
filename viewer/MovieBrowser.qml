
import QtQuick 1.0

Item {
    width: cams.width
    height: cams.height + controls.height + cont_replay_disp.height + scene_title.height
    
    property string scene_template
    property alias first_frame: controls.first_frame
    property alias last_frame: controls.last_frame
    property alias frame_rate: controls.frame_rate
    
    function pad(n, width, z) {
        z = z || '0';
        n = n + '';
        return n.length >= width ? n : new Array(width - n.length + 1).join(z) + n;
    }

    function load_frame() {
        for (var cam = 0; cam < 4; cam++) {
            cams.images.setProperty(cam, "file",
                scene_template.arg(cam + 1).arg(pad(controls.frame, 4, '0')));
            
            if (targets_disp.checked) {
                cams.images.setProperty(cam, "target_list",
                    image_explorer.image_targets(controls.frame, cam));
            } else {
                cams.images.setProperty(cam, "target_list", null);
            }
        }
    }
    
    Text {
        id: scene_title
        width: parent.width
        text: "Scene: " + scene_template.match( /.*\// )
    }

    Multicam {
        id: cams
        anchors.top: scene_title.bottom
    }
    MovieControl {
        id: controls
        anchors.left: cams.left
        anchors.right: cams.right
        anchors.top: cams.bottom
        replay: cont_replay_disp.checked
    }
    
    OptionBox {
        id: cont_replay_disp
        text: "Continuous replay"
        
        anchors.left: cams.left
        anchors.top: controls.bottom
    }
    
    OptionBox {
        id: targets_disp
        text: "Show targets"
        
        anchors.left: cont_replay_disp.right
        anchors.top: controls.bottom
        
        onCheckedChanged: load_frame()
    }
    
    Component.onCompleted: {
        controls.frameChanged.connect(load_frame)
    }
}

